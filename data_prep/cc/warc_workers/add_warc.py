import os 
import glob
import warcio
from tqdm import tqdm

# Find all files in the directory that end with "warc.warc.gz"
directory = '/mnt/nfs/Users/lfsm/wiki_data_2'
files = glob.glob(directory + '/*warc.warc.gz')

from warcio.warcwriter import WARCWriter
output_file = '/home/lfsm/code/RedPajama-Data/data_prep/cc/combined.warc.gz'

with open(output_file, 'ab') as output:
    writer = WARCWriter(output, gzip=True)
    # Iterate over the input files
    for file in tqdm(files):
        # Open each input file for reading in binary mode
        with open(file, 'rb') as input:
            # Iterate over the records in the input file
            for record in warcio.archiveiterator.ArchiveIterator(input):
                # Write the record to the output file
                try:
                    writer.write_record(record)
                except:
                    continue