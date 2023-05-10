import os

os.system(
    "python indexing/subset_data.py create-overlapping-subsets"
)

# Run index_opt.py command
os.system("python indexing/index.py proc_data/train_trec/ --output-folder ./index/")
