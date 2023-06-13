import os 
import glob
import warcio

# Find all files in the directory that end with "warc.warc.gz"
# directory = '/mnt/nfs/Users/lfsm/wiki_data_2'
# files = glob.glob(directory + '/*warc.warc.gz')
files = ['/mnt/nfs/Users/lfsm/wikireference/wikireference.warc.gz']

# Print the names of the matching files
total_pages = 0
for file in files:
    print(f'file is {file}')
    page_count = 0
    # Open the WARC file for reading
    with open(file, 'rb') as warc_file:
        # Iterate over the records in the WARC file
        for record in warcio.archiveiterator.ArchiveIterator(warc_file):
            # Check if the record is a response and has a valid URL
            if record.rec_type == 'response':
                page_count += 1
            if page_count % 1e4 == 0:
                print(f'get {page_count} pages now')
    total_pages += page_count
    print(f'this has {page_count} pages')
print(f'Total pages num is {total_pages}')
