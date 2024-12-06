from gliner import GLiNER
from math import ceil
import numpy as np
import os
from os.path import dirname
import sys

working_path = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
sys.path.append(dirname(os.path.join(working_path, 'model')))
sys.path.append(dirname(os.path.join(working_path, 'checkpoint')))

from model.tags import cult_tags
from model.utils import merge_entities


# upload model
gliner = GLiNER.from_pretrained(os.path.join(working_path, 'checkpoint'), load_tokenizer=True, local_files_only=True)
gliner.eval()


def predict_tokens(text: str, tags: list[str]):
    all_tokens = []
    tokens = []
    start_token_idx_to_text_idx = []
    end_token_idx_to_text_idx = []
    for token, start, end in gliner.token_splitter(text):
        tokens.append(token)
        start_token_idx_to_text_idx.append(start)
        end_token_idx_to_text_idx.append(end)
    all_tokens.append(tokens)

    input_x = [{"tokenized_text": tk, "ner": None} for tk in all_tokens]
    x = gliner.collate_fn(input_x, tags)
    local_scores = gliner.compute_score_eval(x, device='cpu')
    
    return local_scores.squeeze()

def get_token_mapping(text: str):
    return gliner.token_splitter(text)


def model_api(text: str):
    total_markup = []
    for i in range(ceil(len(cult_tags) // 5)):
        tgs = cult_tags[i*5:(i+1)*5]
        scores = predict_tokens(text, tgs)
        total_markup.append(scores.detach().numpy())
        
        if i > 2:
            break
        
    markup = []
    mapping = get_token_mapping(text)
    total_scores = np.concatenate(total_markup, axis=-1)
    for mp, sc in zip(mapping, total_scores):
        markup.append({
            'text': mp[0],
            'start': mp[1],
            'end': mp[2],
            'scores': {cult_tags[i]: float(sc[i]) for i in range(sc.shape[0])}
        })

    return markup
