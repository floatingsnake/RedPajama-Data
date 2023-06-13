import os
import random
from transformers import AutoTokenizer 
from tqdm import tqdm

# num_lines = 100000 
out_file = r'/mnt/nfs/Users/lfsm/wt_ja_corpus/wt_wikicc_ja.txt'
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-roberta-base")

# wiki
dp = r'/mnt/nfs/Users/lfsm/wt_ja_corpus/wt_wikireference.txt'
with open(dp,'r') as f:
    fc = f.read().splitlines()
## length filter
wiki_lines = [i+'\n' for i in fc if len(i)>300]

# cc
cc_dir = r'/mnt/nfs/Users/lfsm/ja_corpus/cc'
cc_names = os.listdir(cc_dir)
cc_files = [os.path.join(cc_dir,name) for name in cc_names]
cc_lines = []
for cc_file in tqdm(cc_files):
    with open(cc_file,'r') as f:
        fc = f.read().splitlines()
        for line in fc:
            ## tokenize
            line = ' '.join(tokenizer.tokenize(line[12:]))
            cc_lines.append('__label__cc '+line+'\n')
            
# write to out_file
print(f'Total wiki lines is {len(wiki_lines)}, cc lines is {len(cc_lines)}')
num_lines = min(len(wiki_lines),len(cc_lines))
print(f'Lines per label is {num_lines}')
wiki_lines = random.sample(wiki_lines,num_lines)
cc_lines = random.sample(cc_lines,num_lines)
with open(out_file, 'w') as f:
    for i in wiki_lines:
        f.write(i)
    for i in cc_lines:
        f.write(i)
