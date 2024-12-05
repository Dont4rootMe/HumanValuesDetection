from torch import nn
import torch

class TextValueExtractor(nn.Module):
    def __init__(self, bert, tokenizer, size=768, max_len=4096, device='cuda', num=112 * 3):
        super().__init__()
        self.tokenizer = tokenizer # tokenizer
        self.bert = bert # bert model
        self.bert.to(device)
        self.device = device # cuda or cpu
        self.max_len = max_len 
        self.size = size
        self.num = num # number of predicted classes
        self.fc1 = nn.Linear(size, num) # classifier
        self.att = nn.Linear(size, num, bias=False) # attention linear layer
        
    def forward(self, texts):
        tokenized = self.tokenizer(texts, return_offsets_mapping=True, 
                                                                   return_tensors="pt", truncation=True, 
                                                                   max_length=self.max_len, padding='max_length').to(self.device)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        offset_mapping = tokenized['offset_mapping'] # begins and ends of tokens in text
        res = self.bert(input_ids, attention_mask)[0] # get token embeddings
        attention_mask[:,0] = 0 # don't consider [CLS] token but ordinary tokens
        token_res = (self.fc1(res) * attention_mask.unsqueeze(-1)).permute(0, 2, 1).reshape(len(texts), self.num, self.max_len, 1) 
        # token logits
        res = nn.functional.softmax(self.att(res) * attention_mask.unsqueeze(-1) + 1e20 * (attention_mask.unsqueeze(-1) - 1), 
                                    dim=1).permute(0, 2, 1).reshape(len(texts), self.num, 1, self.max_len) @ token_res 
        # get text logit by token logits using attention
        return res.reshape(len(texts), self.num), token_res.permute(0, 2, 1, 3).reshape(len(texts), self.max_len, self.num), offset_mapping
