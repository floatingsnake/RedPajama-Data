import os
import torch
from datasets import load_dataset
import random
from transformers import AutoTokenizer 
from tqdm import tqdm

num_lines = 25000
out_file = r'/mnt/nfs/Users/lfsm/wt_ja_corpus/wt_wikioscar.txt' 
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-roberta-base")

# wiki
dp = r'/mnt/nfs/Users/lfsm/wt_ja_corpus/wt_wikireference.txt'
with open(dp,'r') as f:
    fc = f.read().splitlines()
## length filter
fc = [i+'\n' for i in fc if len(i)>300]
wiki_lines = random.sample(fc,num_lines)

# oscar 
oscar_lines = []
cache_dir = "/mnt/nfs/Users/lfsm/oscar"
oscar_ja = load_dataset("oscar", "unshuffled_deduplicated_ja", split='train', streaming=True)
dataset = oscar_ja 
for idx, d in enumerate(dataset):
    if idx >= num_lines:
        break
    try:
        text = d['text'].splitlines()
        text = ' '.join(text)
        tokens = ' '.join(tokenizer.tokenize(text))
        oscar_lines.append('__label__cc '+tokens+'\n') 
    except:
        continue    
# write to out_file
with open(out_file, 'w') as f:
    for i in wiki_lines:
        f.write(i)
    for i in oscar_lines:
        f.write(i)
