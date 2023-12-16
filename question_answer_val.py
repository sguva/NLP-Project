#%%
import time
start_time = time.time()

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import sys
import torch
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel

train_or_val = "val"

DATASET_PATH = os.environ['DATASET_PATH']
SAVE_DIR = os.environ['SAVE_DIR']

SAVE_AG_DIR = os.path.join(SAVE_DIR, "answer_generation")
os.makedirs(SAVE_AG_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(SAVE_AG_DIR, "BLIP2_LoRA")
# INFERENCE_RESULTS_FILE = os.path.join(SAVE_AG_DIR, "answer_generation_results.csv")
INFERENCE_RESULTS_FILE = os.path.join("answer_generation_results.csv")

PROC_LIST = os.sched_getaffinity(0)
print(f"Available processors list: {PROC_LIST}")
NUM_PROC = len(PROC_LIST)

#%% Parameters

# Number of files to load. -1 means all the samples
NUM_FILES = -1 

NUM_DATALOADER_WORKERS = NUM_PROC

#%% Load the data
from data_loading_utils import get_data_encoded_ga, get_data_dic

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

data_encoded, save_results_dic = get_data_encoded_ga(processor, train_or_val, num_files=NUM_FILES)

#% Create PyTorch Dataset
from torch.utils.data import Dataset, DataLoader

class VQADataset(Dataset):
    def __init__(self, data_encoded):
        self.data_encoded = data_encoded

    def __len__(self):
        return len(self.data_encoded)

    def __getitem__(self, idx):
        return self.data_encoded[idx]

def collate_fn(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch

val_dataset = VQADataset(data_encoded)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn, 
                            pin_memory=False, num_workers=NUM_DATALOADER_WORKERS)

#%% Load model and run inference
print("\n Loading the model..........")

peft_model_id = MODEL_SAVE_PATH
config = PeftConfig.from_pretrained(peft_model_id)
model = Blip2ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, device_map="auto")
model = PeftModel.from_pretrained(model, peft_model_id)
# type(model): peft.peft_model.PeftModel

model.eval()

generated_texts = []

# tqdm to display a progress bar
progress_bar = tqdm(val_dataloader, desc=f"Running Inference:")
for idx, batch in enumerate(progress_bar):    

    input_ids = batch["input_ids"].to(device)
    pixel_values = batch["pixel_values"].to(device, torch.float16)
    attention_mask = batch["attention_mask"].to(device)
    # print(input_ids)
    generated_ids = model.generate(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, max_new_tokens=50)
    # print(generated_ids)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    generated_texts.append(generated_text)

#%% Save to csv file

del save_results_dic["image_path"]
save_results_dic["generated_answer"] = generated_texts

# Create a DataFrame from the dictionary
df = pd.DataFrame(save_results_dic)

# Save the DataFrame to a CSV file
df.to_csv(INFERENCE_RESULTS_FILE, index=False)
print(f"---------Saved the results at {INFERENCE_RESULTS_FILE}-----------")

# %%
from datetime import timedelta

get_hhmmss = lambda secs: str(timedelta(seconds=secs))

end_time = time.time()
total_time = end_time - start_time

# print(f"Start Time: {get_hhmmss(start_time)}")
# print(f"End time: {get_hhmmss(end_time)}")
print(f"Total time taken: {get_hhmmss(total_time)}")
# %%

