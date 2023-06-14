
from tqdm import tqdm
import fasttext
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import os
import multiprocessing as mp
import matplotlib.pylab as plt

def count_one_file(input_path):
    df = pd.read_parquet(input_path)
    num_tokens = df['tokens'].apply(lambda x: len(x.split(' '))).sum()
    return num_tokens

if __name__ == '__main__':
    # input_dir = '/mnt/nfs/Users/lfsm/mc4/mc4/ja/0.0.0/99acea4a740b4cc36e4a93a238c7de11b0ce341d65b7d37168b3b90fd64721d2/'
    input_dir = '/mnt/nfs/Users/lfsm/filterd_mc4/cc_filter_thd0.2'
    input_files = [os.path.join(input_dir,name) for name in os.listdir(input_dir)]
    pool_size = mp.cpu_count()
    num_tokens = []
    for input_file in tqdm(input_files):
        num_tokens.append(count_one_file(input_file))
    # with mp.Pool(pool_size) as p:
    #     num_tokens = p.map(count_one_file, input_files)
    total = 0
    for num in num_tokens:
        total += num
    print(f'total number tokens of filted mc4 is {total/1e9} B')
    # mc4-train-00545-of-01645.parquet 2139618 5.2M 
    # 1GB -> 0.36B tokens
    # 100B -> 200GB