import pandas as pd
import matplotlib.pylab as plt
out_file = '/home/lfsm/code/RedPajama-Data/data_prep/cc/result/huu_mc4ja_sample.parquet'

df = pd.read_parquet(out_file)
plt.hist(df['wiki_prob'],bins=50)

def check_short_line(df,thd=200):
    total=0
    count=0
    for i in df['tokens']:
        total += 1
        if len(i) < thd:
            count += 1
    print(f'Total lines {total} short {count} percnet{count/total*100:.2f} %')

def check_length(df,max_length=10000):
    name = 'tokens'
    if name not in df.columns:
        name = 'text'
    if 'length' not in df.columns:
        df['length'] = df[name].apply(len)
    plt.hist(df[df['length']<max_length]['length'],bins=50)

def list2df(lines):
    df = pd.DataFrame(lines,columns=['tokens'])
    df['length'] = df['tokens'].apply(len)
    return df

f_path = r'/home/lfsm/code/RedPajama-Data/data_prep/cc/result/wt_wikicc_res.parquet'
df = pd.read_parquet(f_path)
pos = df[df['wiki_prob']>0.2]
neg = df[df['wiki_prob']<0.2]

pos_sample = pos.sample(n=1800)
neg_sample = neg.sample(n=1800)

