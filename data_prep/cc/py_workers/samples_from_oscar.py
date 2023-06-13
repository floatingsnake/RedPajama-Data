import torch
from datasets import load_dataset
from tqdm import tqdm

cache_dir = "/mnt/nfs/Users/lfsm/oscar"
oscar_ja = load_dataset("oscar-corpus/OSCAR-2301", "ja", cache_dir=cache_dir)
dataset = oscar_ja['train']
oscar_lines = []
import random
values = random.sample(range(1, 47118202-1000),5000)
f = open('/home/lfsm/code/RedPajama-Data/data_prep/cc/data/oscarsamples_ja.txt', 'w')
for i in tqdm(values):
    try:
        text = dataset.__getitem__(i)['text'].splitlines()
        text = ' '.join(text)
        f.write(text+'\n')  
    except:
        continue    
f.close()