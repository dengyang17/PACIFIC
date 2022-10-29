import os
import logging
import torch
import pickle
import json

logger = logging.getLogger(__name__)

def write_pkl(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_and_cache_examples(args, tokenizer, evaluate=False):
    mode = args.set_name if evaluate else 'train'
    print(mode)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'unipcqa_{}_{}_{}_{}_{}'.format(
        args.data_name,
        mode,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(args.max_target_length)))

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = read_pkl(cached_features_file)
        print("Loaded number of instance:", len(features['source_ids']))
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        features = convert_to_features(args, tokenizer, mode)
        print("Loaded number of instance:", len(features['source_ids']))
    
        logger.info("Saving features into cached file %s", cached_features_file)
        write_pkl(features, cached_features_file)
    return features

def convert_to_features(args, tokenizer, mode):
    path = os.path.join(args.data_dir, '{}/{}.json'.format(args.data_name, mode))
    print('tokenizing {}'.format(path))
    #print(tokenizer.SPECIAL_TOKENS_ATTRIBUTES)
    with open(path, 'r', encoding='utf-8') as infile:
        max_dia_len = 0
        avg_dia_len = []
        max_res_len = 0
        avg_res_len = []
        max_resp_len = 0
        max_know_len = 0
        source_ids = []
        target_ids = []

        data = json.load(infile)

        for sample in data:
            table = sample['table']['table']
            paras = sample['paragraphs']
            dial = sample['questions']
            source_id = []
            target_id = []

            table_text = []
            for row in table:
                table_text.append("{} : ".format(row[0]) + " | ".join(row[1:]))

            para_text = []
            for para in paras:
                text = para['text'].replace('\n', ' ')
                para_text.append(text)

            for turn in dial:
                source_id += tokenizer.encode('[user]' + turn['question'])
                req_clari = False
                if args.data_name == 'findial':
                    if turn['req_clari']:
                        req_clari = True
                if req_clari:
                    target_goal = tokenizer.encode('[goal]' + 'True')[:-1]
                    target_rewq = tokenizer.encode('[new_q]' + turn['original_question'])[:-1]
                    target_resp = tokenizer.encode('[system]' + str(turn['answer']))
                else:
                    target_goal = tokenizer.encode('[goal]' + 'False')[:-1]
                    target_rewq = tokenizer.encode('[new_q]' + turn['original_question'])[:-1]
                    if turn['answer_type'] in ['span', 'multi-span']:
                        answer = str(turn['answer'])
                    elif turn['answer_type'] == 'count':
                        answer = "len(" + str(turn['derivation'].split('##')) + ')'
                    else:
                        answer = turn['derivation']
                    target_resp = tokenizer.encode('[system]' + answer + '[scale]' + turn['scale'])
                
                #target_id = target_goal + target_rewq + target_resp
                target_id = target_goal + target_resp

                new_source_id = tokenizer.encode('[paragraph]' + ' '.join(para_text)) + tokenizer.encode('[table]' + '\n'.join(table_text)) + source_id

                source_id += target_resp

                source_ids.append(new_source_id[-args.max_seq_length+1:])
                target_ids.append(target_id[:args.max_target_length])

                avg_dia_len.append(len(new_source_id))
                max_dia_len = max(max_dia_len, len(new_source_id))
                avg_res_len.append(len(target_id))
                max_res_len = max(max_res_len, len(target_id))

        print('{} set, max_res_len: {}, max_dia_len: {}, avg_res_len: {}, avg_dia_len: {}'.format(mode, max_res_len, max_dia_len, float(sum(avg_res_len))/len(avg_res_len), float(sum(avg_dia_len))/len(avg_dia_len)))
    
    return {'source_ids':source_ids, 'target_ids':target_ids}

