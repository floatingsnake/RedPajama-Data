
from tqdm import tqdm
import fasttext
model = fasttext.load_model("/home/lfsm/code/RedPajama-Data/data_prep/cc/model/wt_wikioscar_ja.bin")

import pandas as pd
import matplotlib.pylab as plt
out_file = '/home/lfsm/code/RedPajama-Data/data_prep/cc/result/oscar_wikioscar_ja.parquet'

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-roberta-base")

data = {'label':[],'wiki_prob':[],'tokens' :[], 'length' :[]}
df = pd.DataFrame(data)
dp = '/home/lfsm/code/RedPajama-Data/data_prep/cc/data/oscarsamples_ja.txt'
with open(dp,'r') as f:
    fc = f.read().splitlines()
for text in tqdm(fc):
    tokens = ' '.join(tokenizer.tokenize(text))
    # tokens = text 
    pred = model.predict(tokens)
    (pred_label, pred_prob) = pred
    wiki_prob = pred_prob[0]
    if pred_label[0] == "__label__cc":
        wiki_prob = 1 - wiki_prob
    new_row = pd.DataFrame({'label':[pred_label], 'wiki_prob': [wiki_prob],'tokens':[tokens], 'length':[len(tokens)]})
    df = pd.concat([df,new_row],ignore_index=True)
df.to_parquet(out_file)  