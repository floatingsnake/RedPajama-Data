from tqdm import tqdm
import multiprocess as mp
from transformers import AutoTokenizer 
import os 
import argparse
import psutil 

def check_memory():
    # Get the memory information
    memory = psutil.virtual_memory()
    # Print the total, available, and used memory
    print(f"Used memory: {memory.used / (1024**2)} MB")

def tokenize_file(f_path,out_dir):
    out_file_name = 'wt_'+os.path.basename(f_path)
    out_file = os.path.join(out_dir,out_file_name)
    # tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-roberta-base")
    with open(f_path,'r') as f:
        fc = f.read().splitlines()
    wt_lines = []
    for idx in tqdm(range(len(fc))):
        line = fc[idx]
        # for line in fc:
        if line.startswith('__label__wiki') and len(line)>15:
            line = ' '.join(tokenizer.tokenize(line[14:]))
            wt_lines.append('__label__wiki '+line+'\n')    
        elif line.startswith('__label__cc') and len(line)>13:
            line = ' '.join(tokenizer.tokenize(line[12:]))
            wt_lines.append('__label__cc '+line+'\n')  
        else:
            continue  

    with open(out_file,'w') as f:
        for l in wt_lines:
            f.write(l)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        help="path to txt file",
        default="/home/lfsm/code/RedPajama-Data/data_prep/cc/data/wikivsmc4_ja.txt"
    )

    parser.add_argument(
        "--outdir",
        "-o",
        help="dir path to the transformed file",
        default="/mnt/nfs/Users/lfsm/wt_ja_corpus"
    )

    args = parser.parse_args()
    return args

def main():
    args = get_parser()
    f_path = args.input
    out_dir = args.outdir
    tokenize_file(f_path,out_dir)

if __name__ == '__main__':
    main()