import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PretrainedConfig, PreTrainedModel
from math import ceil
import numpy as np
import os
from os.path import dirname
import sys

working_path = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
sys.path.append(dirname(os.path.join(working_path, 'model')))

from model.cult_tags import group_names, tags_list, borders

class HVDConfig(PretrainedConfig):
    def __init__(self, size=768, max_len=4096, device='cpu', num=112 * 3, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.max_len = max_len
        self.device = device
        self.num = num

class HVDModel(PreTrainedModel):
    config_class = HVDConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.size = config.size
        self.max_len = config.max_len
        self.num = config.num
        self.bert = AutoModel.from_pretrained(os.path.join(working_path,'model/bert/bert_transformer'))
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(working_path, 'model/bert/bert_tokenizer'))
        self.fc1 = nn.Linear(self.size, self.num) # classifier
        self.att = nn.Linear(self.size, self.num, bias=False) # attention linear layer
        
    def forward(self, texts):
        tokenized = self.tokenizer(texts, return_offsets_mapping=True, 
                                                                   return_tensors="pt", truncation=True, 
                                                                   max_length=self.max_len, padding='max_length').to(self.bert.device)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        offset_mapping = tokenized['offset_mapping'] # begins and ends of tokens in text
        res = self.bert(input_ids, attention_mask)[0] # get token embeddings
        attention_mask[:,0] = 0 # don't consider [CLS] token but ordinary tokens
        token_res = (self.fc1(res) * attention_mask.unsqueeze(-1)).permute(0, 2, 1).reshape(len(texts), self.num, self.max_len, 1) 
        # token logits
        res = nn.functional.softmax(self.att(res) * attention_mask.unsqueeze(-1) + 1e20 * (attention_mask.unsqueeze(-1) - 1), 
                                    dim=1).permute(0, 2, 1).reshape(len(texts), self.num, 1, self.max_len) @ token_res 
        # get text logit by token logits using attention pooling
        return res.reshape(len(texts), self.num), token_res.permute(0, 2, 1, 3).reshape(len(texts), self.max_len, self.num), offset_mapping

# upload model
model = HVDModel.from_pretrained(os.path.join(working_path, 'model/bert/bert_checkpoint'))
model.to('cpu')
model.eval()

def model_api(text: str):
    with torch.no_grad():
        res, token_res, offset_mapping = model([text])
    res = res[0].reshape(3, 112)
    token_res = token_res[0].reshape(4096, 3, 112)
    offset_mapping = offset_mapping[0]
    for i in [1, 2]: # tonality fix
        res[i] = torch.min(res[0], res[i])
        token_res[:,i] = torch.minimum(token_res[:,0], token_res[:,i])
    for j in range(7):
        res[:,105 + j] = torch.maximum(torch.max(res[:,borders[j]:borders[j + 1]], dim=1).values, res[:,105 + j])
        token_res[:,:,105 + j] = torch.maximum(torch.max(token_res[:,:,borders[j]:borders[j + 1]], dim=2).values, token_res[:,:,105 + j])
    markup = {
        'text': {
            'high': [],
            'low': []
        },
        'tokens': []
    }
    for i in range(7):
        markup['text']['high'].append({
            'name': group_names[i],
            'score': nn.functional.sigmoid(res[0,105 + i]).item(),
            'positive': nn.functional.sigmoid(res[1,105 + i]).item(),
            'negative': nn.functional.sigmoid(res[2,105 + i]).item()
        })
    for i in range(105):
        markup['text']['low'].append({
            'name': tags_list[i],
            'score': nn.functional.sigmoid(res[0,i]).item(),
            'positive': nn.functional.sigmoid(res[1,i]).item(),
            'negative': nn.functional.sigmoid(res[2,i]).item()
        })
    for j in range(tve.max_len):
        if offset_mapping[j, 0] < offset_mapping[j, 1]:
            token_markup = {
                'token_text': text[offset_mapping[j, 0]:offset_mapping[j, 1]],
                'begin': offset_mapping[j, 0],
                'end': offset_mapping[j, 1],
                'high': [],
                'low': []
            }
            for i in range(7):
                token_markup['high'].append({
                    'name': group_names[i],
                    'score': nn.functional.sigmoid(token_res[j, 0,105 + i]).item(),
                    'positive': nn.functional.sigmoid(token_res[j, 1,105 + i]).item(),
                    'negative': nn.functional.sigmoid(token_res[j, 2,105 + i]).item()
                })
            for i in range(105):
                token_markup['low'].append({
                    'name': tags_list[i],
                    'score': nn.functional.sigmoid(token_res[j,0,i]).item(),
                    'positive': nn.functional.sigmoid(token_res[j,1,i]).item(),
                    'negative': nn.functional.sigmoid(token_res[j,2,i]).item()
                })
            markup['tokens'].append(token_markup)
    return markup
