import multiprocessing
import subprocess
import time
import os

# Define the command to run
output_dir = "/mnt/nfs/Users/lfsm/wiki_data_3/"
txt_file = '200k_2.txt'
num_processes = multiprocessing.cpu_count()

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
command = "wget --timeout=5 --tries=2 -O /dev/null"
# Define a function that downloads a batch of URLs
def download_urls(p_idx, start, end):
    # Extract the subset of URLs to download
    with open(txt_file) as f:
        urls = f.readlines()[start:end]
    tmp_txt = f'{output_dir}mp_{p_idx:02d}_urls.txt' 
    with open(tmp_txt, "w") as f:
        for url in urls:
            f.write(str(url))
    output = f'--warc-file={output_dir}mp_{p_idx:02d}_wiki_reference_pages.warc'
    in_urls = f'--input-file={tmp_txt}' 
    # Run the command for each URL
    # for url in urls:
    subprocess.run(command.split() +[output.strip()] + [in_urls])
    os.remove(tmp_txt)

# Launch multiple processes to download the URLs in parallel
with open(txt_file) as f:
    num_urls = len(f.readlines())

processes = []
for i in range(num_processes):
    start = i * (num_urls // num_processes)
    end = (i + 1) * (num_urls // num_processes) if i < num_processes - 1 else num_urls
    p = multiprocessing.Process(target=download_urls, args=(i, start, end))
    p.start()
    time.sleep(1)
    processes.append(p)

# Wait for all processes to finish
for p in processes:
    p.join()
