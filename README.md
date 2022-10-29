PACIFIC: Towards Proactive Conversational Question Answering over Tabular and Textual Data in Finance
====================

**PACIFIC** (**P**ro**A**ctive **C**onversat**i**onal Question Answering in **FI**nan**C**e) contains 2,757 dialogues associated with 1,9008 QA turns. 

You can download our PACIFIC dataset via [PACIFIC dataset](https://github.com/dengyang17/PACIFIC/tree/master/data).
                
For more information, please refer to our [PACIFIC website](https://nextplusplus.github.io/PACIFIC/) (coming soon) or read our EMNLP 2022 paper [PDF](https://arxiv.org/abs/2210.08817).


## UniPCQA Model

### Training & Testing
`python train.py --do_train --do_eval --max_seq_length=1280 --max_target_length=128 --gpu=<your_gpu_ids> --overwrite_output_dir --per_gpu_train_batch_size=<your_batch_size> --per_gpu_eval_batch_size=<your_batch_size> --model_name_or_path="Salesforce/codet5-base" --data_name='pacific' --model_name='codet5'`


__Please kindly cite our work if you use our dataset or codes, thank you.__
```bash
@inproceedings{emnlp22-pacific,
  author    = {Yang Deng and
               Wenqiang Lei and
               Wenxuan Zhang and
               Wai Lam and
               Tat{-}Seng Chua},
  title     = {{PACIFIC:} Towards Proactive Conversational Question Answering over
               Tabular and Textual Data in Finance},
  booktitle = {Proceedings of the 2022 Conference on Empirical Methods in Natural
               Language Processing, {EMNLP} 2022},
  year      = {2022},
}
```
