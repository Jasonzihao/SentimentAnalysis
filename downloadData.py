from datasets import Dataset, load_dataset, load_from_disk
from datasets import config
print(config.HF_DATASETS_CACHE)
dataset = load_dataset("imdb")
# dataset.save_to_disk("imdb") # 保存到该目录下
