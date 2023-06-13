import pandas as pd
from datasets import load_dataset
import torch
from tqdm import tqdm

dp = r'/home/lfsm/code/RedPajama-Data/data_prep/cc/data/wikivsmc4_ja.txt'
out_file =r'/home/lfsm/code/RedPajama-Data/data_prep/cc/data/wikivsmc4oscar_ja.txt' 

with open(dp,'r') as f:
    fc = f.read().splitlines()

wiki_lines = []
mc_lines = []
oscar_lines = []
for line in fc:
    if line.startswith('__label__wiki'):
        wiki_lines.append(line)
    elif line.startswith('__label__cc'):
        mc_lines.append(line)
    else:
        print('what this line is ?')         

mc_lines = mc_lines[:int(len(mc_lines)/2)]


cache_dir = "/mnt/nfs/Users/lfsm/oscar"
oscar_ja = load_dataset("oscar-corpus/OSCAR-2301", "ja", cache_dir=cache_dir)
dataset = oscar_ja['train']

import random
values = random.sample(range(1, 47118202-1000),len(mc_lines))
for i in tqdm(values):
    try:
        text = dataset.__getitem__(i)['text'].splitlines()
        oscar_lines.append("__label__cc " +" ".join(text))  
    except:
        continue
    
with open(out_file,'w') as of:
    for i in wiki_lines:
        of.write(i+'\n') 
    for i in mc_lines:
        of.write(i+'\n') 
        # 50716
    for i in oscar_lines:
        of.write(i+'\n') 

print('I am done')