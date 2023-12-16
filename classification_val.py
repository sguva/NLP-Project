#%% Load the saved classification model and run it on validation data and save the results to csv file.
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

from transformers import ViltProcessor, ViltForQuestionAnswering, pipeline


train_or_val = "val"

DATASET_PATH = os.environ['DATASET_PATH']
SAVE_DIR = os.environ['SAVE_DIR']
CSV_FILE_PATH = os.path.join(SAVE_DIR, f"data_generation/fitlered_data_with_position_{train_or_val}.csv")

SAVE_CLS_DIR = os.path.join(SAVE_DIR, "classification")

MODEL_SAVE_PATH = os.path.join(SAVE_CLS_DIR, "ViLT")
PIPELINE_SAVE_PATH = os.path.join(SAVE_CLS_DIR, "ViLT_Pipeline")
# INFERENCE_RESULTS_FILE = os.path.join(SAVE_CLS_DIR, "classification_results.csv")
INFERENCE_RESULTS_FILE = os.path.join("classification_results.csv")

#%% Parameters

# Batch size for dataset map method.
DSMAP_BATCH_SIZE = 200
PROC_LIST = os.sched_getaffinity(0)
print(f"Available processors list: {PROC_LIST}")
NUM_PROC = len(PROC_LIST)

# Number of files to load. -1 means all the samples
NUM_FILES = -1
# NUM_FILES = BATCH_SIZE*5

#%% Dictionaries that maps the label name to an integer and vice versa:

ids = list(range(0,9))
id2label = {0: "topLeft", 1: "topCenter", 2: "topRight", 3: "middleLeft", 4: "middleCenter", 5: "middleRight", 6: "bottomLeft", 7: "bottomCenter", 8: "bottomRight"}
labels = list(id2label.values())
label2id = {label: idx for idx, label in enumerate(labels)}

#%%  Load the data
from data_loading_utils import get_data_dic_cls
from datasets import load_dataset, load_from_disk, Dataset

print("Loading dictionary............")
dic = get_data_dic_cls(CSV_FILE_PATH, train_or_val=train_or_val, num_files=NUM_FILES)
print(f"Len of dic: {len(dic['image_id'])}")

dataset = Dataset.from_dict(dic)
print(dataset)
#%% Load the saved model

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
pipe = pipeline("visual-question-answering", model=PIPELINE_SAVE_PATH, device=device)


#%%
def preprocess_data(examples):
    image_paths = examples['image_path']
    images = [Image.open(image_path).convert('RGB').resize((384, 384)) for image_path in image_paths]

    out = {}
    out["image"] = images
    out["question"] = examples['categoryName'] 
    out["answer"] = examples['positionName']    

    return out

#%% Apply the preprocessing function over the entire dataset using Datasets map function
processed_dataset = dataset.map(preprocess_data, batched=True, batch_size=DSMAP_BATCH_SIZE,
                                remove_columns=["row_id", "image_id", "image_path", "positionName", "categoryName"], 
                                keep_in_memory=True, num_proc=NUM_PROC)
print("Processed dataset:")
print(processed_dataset)

#%%
predictedPosition = []
actualPosition = []
progress_bar = tqdm(processed_dataset, desc=f"Running Inference:")
for idx, sample in enumerate(progress_bar): 
    pred = pipe(sample, top_k=1)[0]
    # print(f"pred: {pred} - actual: {sample['answer']}")
    actualPosition.append(sample['answer'])
    predictedPosition.append(pred['answer'])

acc = np.mean(np.equal(actualPosition, predictedPosition))
print(f"Validation Accuracy: {acc}")

# %% Save results to disk.

save_results = dic.copy()
del save_results["image_path"]
save_results["predictedPosition"] = predictedPosition
# save_results["actualPosition"] = actualPosition
# Create a DataFrame from the dictionary
df = pd.DataFrame(save_results)

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