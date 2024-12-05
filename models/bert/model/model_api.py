import torch
from math import ceil
import numpy as np
import os
from os.path import dirname
import sys

working_path = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
sys.path.append(dirname(os.path.join(working_path, 'model')))

from value_extractor import TextValueExtractor
from model.tags import group_names, tags_list, borders


# upload model
model = torch.load(open("bert_checkpoint.pt", "rb"))
model.to('cuda')
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
