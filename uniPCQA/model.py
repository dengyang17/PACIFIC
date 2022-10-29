from transformers import T5ForConditionalGeneration
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F


class UniMind(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                    config=config, cache_dir=args.cache_dir)
        self.config = config

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        loss_g = outputs[0]
        return loss_g