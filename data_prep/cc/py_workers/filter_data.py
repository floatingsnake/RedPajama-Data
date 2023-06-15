
from tqdm import tqdm
import fasttext
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import os
import multiprocessing as mp
import matplotlib.pylab as plt


model = fasttext.load_model("/home/lfsm/code/RedPajama-Data/data_prep/cc/model/wt_wikicc_ja.bin")

def filter_one_file(input_path,out_dir='/mnt/nfs/Users/lfsm/filterd_oscar2023/cc_filter_thd0.2'):
    name = os.path.basename(input_path)
    out_file = os.path.join(out_dir,name)
    # out_file to save the result, you can change this to your path
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-roberta-base")
    # tokenize the text before feed to the filter
    data = {'label':[],'wiki_prob':[],'tokens' :[], 'length' :[]}
    df = pd.DataFrame(data)
    # create a dataframe to save the result
    with pa.OSFile(input_path, 'rb') as f:
        reader = pa.ipc.open_stream(f)
        for batch in tqdm(reader):
            df_mc4 = batch.to_pandas()
            df_text = df_mc4['text']
            df_text_list = df_text.tolist()
            for text in df_text_list:
                # do the filter now!
                tokens = ' '.join(tokenizer.tokenize(text))
                # tokenize the text
                pred = model.predict(tokens)
                # feed to model to get the result
                (pred_label, pred_prob) = pred
                wiki_prob = pred_prob[0]
                if pred_label[0] == "__label__cc":
                    wiki_prob = 1 - wiki_prob
                if wiki_prob > 0.2:
                    # get the probility of being a wiki-reference for the text
                    new_row = pd.DataFrame({'label':[pred_label], 'wiki_prob': [wiki_prob],'tokens':[tokens], 'length':[len(tokens)]})
                    df = pd.concat([df,new_row],ignore_index=True)
    df.to_parquet(out_file)  
    # save the data

if __name__ == '__main__':
    # input_dir = '/mnt/nfs/Users/lfsm/mc4/mc4/ja/0.0.0/99acea4a740b4cc36e4a93a238c7de11b0ce341d65b7d37168b3b90fd64721d2/'
    input_dir = '/mnt/nfs/Users/lfsm/oscar-2023/oscar-corpus___oscar-2301/ja/2023.1.0/156efb8ba9f439f881d8f41fd7fddd5e04604bc27505c46ddef015f2fc551a4a'
    # work on oscar now
    input_files = [os.path.join(input_dir,name) for name in os.listdir(input_dir)]
    pool_size = mp.cpu_count()
    with mp.Pool(pool_size) as p:
        p.map(filter_one_file, input_files)

    # mc4-train-00545-of-01645.parquet 2139618 5.2M 
    # 1GB -> 0.4-0.5B tokens
    # 100B -> 200GB