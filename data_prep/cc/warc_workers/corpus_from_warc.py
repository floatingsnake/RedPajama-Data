import os 
import warcio
from tqdm import tqdm
import chardet
from langdetect import detect
from bs4 import BeautifulSoup
from html2text import html2text
import argparse

def dehtml(text):
    text = html2text(text)  # remove any HTML
    return text

def process_warc(args):
    lang_ja = args.out_file
    f_path = args.input
    ja = 0
    en = 0
    other = 0
    with open(lang_ja, "w") as f_ja, open(f_path, 'rb') as warc_file:
        for record in tqdm(warcio.archiveiterator.ArchiveIterator(warc_file)):
            if record.rec_type == 'response':
                try:
                    # bytes
                    bytes = record.content_stream().read()#.decode(encoding="utf-8", errors='ignore')
                    enc = chardet.detect(bytes[:1024*100])['encoding']
                    if enc is None:
                        continue
                    html = bytes.decode(encoding=enc, errors='ignore')
                    soup = BeautifulSoup(html, 'html.parser')
                    text = soup.get_text()
                    # if(len(text) < 300):
                    #     continue
                    lang_flag = detect(text)
                    text=dehtml(text)
                    text = text.splitlines()
                    # text = tokenizer.encode(text) 
                    if lang_flag == 'ja':
                        ja += 1
                        f_ja.write("__label__cc " +" ".join(text)+'\n')
                    elif lang_flag == 'en':
                        en += 1
                    else:
                        other += 1
                except:
                    continue
    print(f'Got {ja+en+other} pages in total: ja is {ja}, en is {en}, other is {other}')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        help="path to warc file",
        default="enwiki-20230401-pages-articles-multistream.xml",
    )

    parser.add_argument(
        "--outdir",
        "-o",
        help="dir path to the transformed file",
        default="/home/lfsm/code/mm_builder/dataset/wiki_en/interleaved",
    )

    args = parser.parse_args()
    return args

def main():
    args = get_parser()
    out_file_name = os.path.basename(args.input).replace('.warc.gz', '.txt')
    args.out_file = os.path.join(args.outdir,out_file_name)
    process_warc(args)

if __name__ == '__main__':
    main()