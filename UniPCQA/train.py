import argparse
import glob
import logging
import os
import random
from transformers import T5Tokenizer, RobertaTokenizer, T5ForConditionalGeneration, T5Config, AdamW
from pytorch_transformers import WarmupLinearSchedule, WEIGHTS_NAME
import torch
import utils
import data_reader
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import metrics
from model import UniPCQA
import math

tok = {'t5': T5Tokenizer, 'codet5': RobertaTokenizer}

class DataFrame(Dataset):
    def __init__(self, data, args):
        self.source_ids = data['source_ids']
        self.target_ids = data['target_ids']
        self.max_len = args.max_seq_length
        self.max_tgt_len = args.max_target_length

    def __getitem__(self, index):
        return self.source_ids[index][:self.max_len], self.target_ids[index][:self.max_tgt_len]
    
    def __len__(self):
        return len(self.source_ids)


def collate_fn(data):
    source_ids, target_ids = zip(*data)
    batch_size = len(source_ids)

    input_ids = [torch.tensor(source_id).long() for source_id in source_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    attention_mask = input_ids.ne(0)
    labels = [torch.tensor(target_id).long() for target_id in target_ids]
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    
    return {'input_ids':  input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            }


def train(args, train_dataset, model, tokenizer):
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, len(args.device_id))
    train_dataloader = DataLoader(DataFrame(train_dataset, args), batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if len(args.device_id) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_id)
        #torch.distributed.init_process_group(backend="nccl")
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.device_id, find_unused_parameters=True)
    
    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logging.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)
    
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    utils.set_seed(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    global_step = 0

    #total_rouge = evaluate(args, model, tokenizer, save_output=True)
    best_f1 = 0 #total_rouge[2]
    
    for e in train_iterator:
        logging.info("training for epoch {} ...".format(e))
        print("training for epoch {} ...".format(e))
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            #batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch['input_ids'].to(args.device),
                        'attention_mask': batch['attention_mask'].to(args.device),
                        'labels': batch['labels'].to(args.device)}
            outputs = model(**inputs)
            loss = outputs#[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if len(args.device_id) > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
            global_step += 1
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
        tb_writer.add_scalar('loss', (tr_loss - logging_loss) / (step+1), global_step)
        print('loss: {}'.format((tr_loss - logging_loss) / (step+1)))
        logging_loss = tr_loss

        # Log metrics
        results = evaluate(args, model, tokenizer, save_output=True)
                    
        if results['cqa'][0] > best_f1:
            # Save model checkpoint
            best_f1 = results['cqa'][0]
            output_dir = os.path.join(args.output_dir, 'best_checkpoint')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model,
                            'module') else model  # Take care of distributed/parallel training
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logging.info("Saving model checkpoint to %s", output_dir)
    
    tb_writer.close()

    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, save_output=False):

    eval_dataset = data_reader.load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, len(args.device_id))

    all_metrics = []
    eval_dataloader = DataLoader(DataFrame(eval_dataset, args), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    # Eval!
    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)
    count = 0
    preds = []
    sample_preds = []
    targets = []

    model_to_eval = model.module if hasattr(model, 'module') else model
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            beam_size = args.beam_size
            max_length = args.max_target_length
            generated_ids = model_to_eval.t5.generate(
                input_ids=batch['input_ids'].to(args.device),
                attention_mask=batch['attention_mask'].to(args.device),
                num_beams=1,
                max_length=max_length,
                #repetition_penalty=2.5,
                #length_penalty=1.5,
                early_stopping=True,
            )
            num_return_sequences = args.num_return_sequences
            sample_generated_ids = model_to_eval.t5.generate(
                input_ids=batch['input_ids'].to(args.device),
                attention_mask=batch['attention_mask'].to(args.device),
                do_sample=True,
                max_length=max_length,
                top_k=40, 
                temperature=0.5, 
                num_return_sequences=num_return_sequences,
                early_stopping=True
            )
            sample_generated_ids = sample_generated_ids.view(-1, num_return_sequences, sample_generated_ids.size(-1))
            
            preds.extend([
                tokenizer.decode(
                    g, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                for g in generated_ids
            ])

            for g in sample_generated_ids:
                sample_pred = [
                tokenizer.decode(
                    b, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                for b in g]
                sample_preds.append(sample_pred)
            
                
            targets.extend([
                tokenizer.decode(
                    g, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                for g in batch['labels']
            ])

    if save_output:
        output_dir = args.output_dir
        with open(os.path.join(args.output_dir, '{}_{}_{}.decoded'.format(args.data_name, args.set_name, list(filter(None, args.model_name_or_path.split('/'))).pop())), 'w') as outfile,\
            open(os.path.join(args.output_dir, '{}_{}_{}_beam.decoded'.format(args.data_name, args.set_name, list(filter(None, args.model_name_or_path.split('/'))).pop())), 'w') as outfile2,\
            open(os.path.join(args.output_dir, '{}_{}.reference'.format(args.data_name, args.set_name)), 'w') as reffile:
            for p, t, b in zip(preds, targets, sample_preds):
                outfile.write("{}\n".format(p))
                reffile.write("{}\n".format(t))
                outfile2.write("{}\n".format(b))

    auto_scores = metrics.calculate(preds, targets, args.data_name, args.set_name, args.model_name)
    logging.info(auto_scores)
    auto_scores_beam = metrics.calculate_beam(sample_preds, targets, args.data_name, args.set_name, args.model_name)
    logging.info(auto_scores_beam)
    print(auto_scores)
    print(auto_scores_beam)
    return auto_scores



def main():
    parser = argparse.ArgumentParser(description="train.py")

    ## Required parameters
    parser.add_argument('--data_name', default='pacific', type=str,
                        help="dataset name")
    parser.add_argument('--set_name', default='validation', type=str,
                        help="dataset split name")
    parser.add_argument('--model_name', default='t5', type=str,
                        help="model name")
    parser.add_argument("--model_name_or_path", default='t5-base',
                        type=str, help="model name or path")
    parser.add_argument("--output_dir", default='output', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--data_dir", default='../data', type=str,
                        help="The data directory.")
    parser.add_argument("--cache_dir", default='/storage/ydeng/bert', type=str,
                        help="The cache directory.")

    ## Other parameters
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--do_lower_case", action='store_false',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=100, type=int,
                        help="The maximum total output sequence length.")
    parser.add_argument('--beam_size', default=1, type=int)
    parser.add_argument('--num_return_sequences', default=40, type=int)
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gpu', default="0 1 2", type=str,
                        help="Use CUDA on the device.")
    parser.add_argument("--per_gpu_train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--num_train_epochs", default=15, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--warmup_steps", default=400, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", default=-1, type=int,
                        help="DDP requirement.")

    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.data_name, args.model_name)
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    logging.basicConfig(level=logging.DEBUG, filename=args.output_dir + '/log.txt', filemode='a')

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    
    # Setup CUDA, GPU & distributed training
    device, device_id = utils.set_cuda(args)
    args.device = device
    args.device_id = device_id

    # Set seed
    utils.set_seed(args.seed)

    config = T5Config.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = tok[args.model_name].from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir)
    tokenizer.add_special_tokens({'additional_special_tokens':['[goal]','[user]','[system]','[table]','[paragraph]','[scale]','[new_q]']})

    train_dataset = data_reader.load_and_cache_examples(args, tokenizer, evaluate=False)
    
    model = UniPCQA(args, config)
    model.t5.resize_token_embeddings(len(tokenizer))

    model.to(args.device)

    logging.info("Training/evaluation parameters %s", args)
    output_dir = os.path.join(args.output_dir, 'best_checkpoint')

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logging.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        tokenizer.save_pretrained(output_dir)

    # Evaluation
    if args.do_eval:
        # Load a trained model and vocabulary that you have fine-tuned
        if hasattr(model, 'module'):
            model.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        else:
            model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        tokenizer = tok[args.model_name].from_pretrained(output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)
        args.set_name = 'test'
        evaluate(args, model, tokenizer, save_output=True)


if __name__ == "__main__":
    main()