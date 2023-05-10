import os

# Run subset-train-data command
os.system(
    "/NIR/.venv/bin/python /NIR/subset_data.py subset-train-data"
)

# Run index_opt.py command
os.system("python index.py proc_data/train_trec/ --output-folder ./index/")
