
from tqdm import tqdm
import fasttext
model = fasttext.load_model("/home/lfsm/code/RedPajama-Data/data_prep/cc/model/wt_wikioscar_ja.bin")
# load the filter

import pandas as pd
import matplotlib.pylab as plt
out_file = '/home/lfsm/code/RedPajama-Data/data_prep/cc/result/mc4_wikioscar_ja.parquet'
# out_file to save the result, you can change this to your path

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-roberta-base")
# tokenize the text before feed to the filter

data = {'label':[],'wiki_prob':[],'tokens' :[], 'length' :[]}
df = pd.DataFrame(data)
# create a dataframe to save the result

dp = '/home/lfsm/code/RedPajama-Data/data_prep/cc/data/mc4samples_ja.txt'
with open(dp,'r') as f:
    fc = f.read().splitlines()
# get our data waitted to be filted. 
# you need to change this to read the data from .arrow files from mc4

for text in tqdm(fc):
    # do the filter now!
    tokens = ' '.join(tokenizer.tokenize(text))
    # tokenize the text
    pred = model.predict(tokens)
    # feed to model to get the result
    (pred_label, pred_prob) = pred
    wiki_prob = pred_prob[0]
    if pred_label[0] == "__label__cc":
        wiki_prob = 1 - wiki_prob
    # get the probility of being a wiki-reference for the text
    new_row = pd.DataFrame({'label':[pred_label], 'wiki_prob': [wiki_prob],'tokens':[tokens], 'length':[len(tokens)]})
    df = pd.concat([df,new_row],ignore_index=True)
df.to_parquet(out_file)  
# save the data