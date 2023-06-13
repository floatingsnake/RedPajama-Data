import torch
from datasets import load_dataset

cache_dir = "/mnt/nfs/Users/lfsm/mc4"
mc4_ja = load_dataset("mc4", "ja", cache_dir=cache_dir)
dataset = mc4_ja["train"]
dataloader = torch.utils.data.DataLoader(dataset,shuffle=True)

idx = 0
f = open('/home/lfsm/code/RedPajama-Data/data_prep/cc/data/mc4samples_ja.txt', 'w')
for batch in dataloader:
    idx += 1
    text = batch['text'][0].splitlines()
    text = ' '.join(text)
    if idx > 5000:
        break
    f.write(text+'\n')
f.close()
    