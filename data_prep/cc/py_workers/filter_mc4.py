
from tqdm import tqdm
import fasttext
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import os
import multiprocessing as mp
import matplotlib.pylab as plt


model = fasttext.load_model("/home/lfsm/code/RedPajama-Data/data_prep/cc/model/wt_wikicc_ja.bin")

def filter_one_file(input_path,out_dir='/mnt/nfs/Users/lfsm/filterd_mc4/cc_filter'):
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
        batches = [batch for batch in reader]
    for batch in tqdm(batches):
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
            # get the probility of being a wiki-reference for the text
            new_row = pd.DataFrame({'label':[pred_label], 'wiki_prob': [wiki_prob],'tokens':[tokens], 'length':[len(tokens)]})
            df = pd.concat([df,new_row],ignore_index=True)
    df.to_parquet(out_file)  
    # save the data

if __name__ == '__main__':
    input_dir = '/mnt/nfs/Users/lfsm/mc4/mc4/ja/0.0.0/99acea4a740b4cc36e4a93a238c7de11b0ce341d65b7d37168b3b90fd64721d2/'
    input_files = [os.path.join(input_dir,name) for name in os.listdir(input_dir)]
    pool_size = mp.cpu_count()
    # global pbar
    # pbar = tqdm(total=len(input_files), desc='Processing', unit='task')
    with mp.Pool(pool_size) as p:
        p.map(filter_one_file, input_files)