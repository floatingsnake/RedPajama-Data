import os
import torch
from datasets import load_dataset
import random
from transformers import AutoTokenizer 

num_lines = 100000
out_file = r'/mnt/nfs/Users/lfsm/wt_ja_corpus/wt_wikimc4.txt' 
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-roberta-base")

# wiki
dp = r'/mnt/nfs/Users/lfsm/wt_ja_corpus/wt_wikireference.txt'
with open(dp,'r') as f:
    fc = f.read().splitlines()
## length filter
fc = [i+'\n' for i in fc if len(i)>300]
wiki_lines = random.sample(fc,num_lines)

# mc4
mc4_lines = []
cache_dir = "/mnt/nfs/Users/lfsm/mc4"
mc4_ja = load_dataset("mc4", "ja", cache_dir=cache_dir)
dataset = mc4_ja["train"]
dataloader = torch.utils.data.DataLoader(dataset,shuffle=True)
idx = 0
for batch in dataloader:
    idx += 1
    text = batch['text'][0].splitlines()
    text = ' '.join(text)
    tokens = ' '.join(tokenizer.tokenize(text))
    mc4_lines.append('__label__cc '+tokens+'\n')  
    if idx > num_lines:
        break

# write to out_file
with open(out_file, 'w') as f:
    for i in wiki_lines:
        f.write(i)
    for i in mc4_lines:
        f.write(i)
