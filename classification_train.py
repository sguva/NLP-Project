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

from transformers import ViltProcessor, ViltForQuestionAnswering, pipeline


train_or_val = "train"

DATASET_PATH = os.environ['DATASET_PATH']
SAVE_DIR = os.environ['SAVE_DIR']

SAVE_CLS_DIR = os.path.join(SAVE_DIR, "classification")
os.makedirs(SAVE_CLS_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(SAVE_CLS_DIR, "ViLT")
PIPELINE_SAVE_PATH = os.path.join(SAVE_CLS_DIR, "ViLT_Pipeline")

PROC_LIST = os.sched_getaffinity(0)
print(f"Available processors list: {PROC_LIST}")
NUM_PROC = len(PROC_LIST)


#%% Parameters

BATCH_SIZE = 32

# Number of files to load. -1 means all the samples
NUM_FILES = -1 
# NUM_FILES = BATCH_SIZE*5

NUM_EPOCHS = 2
SAVE_AFTER_EVERY_EPOCH = False
#%% Dictionaries that maps the label name to an integer and vice versa:

ids = list(range(0,9))
id2label = {0: "topLeft", 1: "topCenter", 2: "topRight", 3: "middleLeft", 4: "middleCenter", 5: "middleRight", 6: "bottomLeft", 7: "bottomCenter", 8: "bottomRight"}
labels = list(id2label.values())
label2id = {label: idx for idx, label in enumerate(labels)}

#%% Load data
from data_loading_utils import get_data_encoded_cls

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
data_encoded, dic = get_data_encoded_cls(processor, train_or_val, num_files=NUM_FILES)

#%%
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
        # Check if the item is a list and convert it to a tensor if necessary
        if isinstance(batch[0][key], list):
            # Convert each sublist to a tensor and then stack
            processed_batch[key] = torch.stack([torch.tensor(example[key]) for example in batch])
        else:
            # If it's already a tensor, stack directly
            processed_batch[key] = torch.stack([example[key] for example in batch])
    return processed_batch

train_dataset = VQADataset(data_encoded)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn, 
                            pin_memory=True, num_workers=NUM_PROC)

# Sanity check
# itr = iter(train_dataloader)
# item = next(itr)
# item.keys()
# # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'pixel_values', 'pixel_mask', 'labels'])

#%% Load the model

model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm", num_labels=len(id2label), id2label=id2label, label2id=label2id)
# model = ViltForQuestionAnswering.from_pretrained(MODEL_SAVE_PATH, num_labels=len(id2label), id2label=id2label, label2id=label2id)

# model.vilt.requires_grad_(False)
model.to(device)

#%% Train the model

# Open a file to log the output
save_training_log_path = os.path.join(SAVE_CLS_DIR, 'training_log.txt')

# Initialize a list to store the loss values
loss_values = []

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
model.train()

with open(save_training_log_path, 'a') as f:
    print(f"\nTraining for {NUM_EPOCHS} epochs.............")
    for epoch in range(NUM_EPOCHS):
        # Initialize the loss for the current epoch
        epoch_loss = []
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)#, torch.float16)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            pixel_mask = batch['pixel_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=labels,
                            token_type_ids=token_type_ids,
                            pixel_mask=pixel_mask,
                            )
            
            loss = outputs.loss
            epoch_loss.append(loss.item())

            # Update the progress bar
            progress_bar.set_postfix({"loss": loss.item()})

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        
        # Store the average loss for the current epoch
        loss_values.append(sum(epoch_loss) / len(epoch_loss))

        # Write to the log file
        f.write(f"Epoch {epoch}, Loss: {loss_values[-1]}\n")
                
        # Print to the terminal
        print(f"Epoch {epoch}, Loss: {loss_values[-1]}")

        # Save model after each epoch
        if SAVE_AFTER_EVERY_EPOCH == True:
            print(f"Saving model at {MODEL_SAVE_PATH}................")
            model.save_pretrained(MODEL_SAVE_PATH)
            print("Done saving.")

# Plot the loss values
plt.figure(figsize=(10, 5))
plt.plot(loss_values, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training loss')
plt.legend()
# plt.show()
plt.savefig(os.path.join(SAVE_CLS_DIR, "training_loss.png"))
plt.close()

print(f"\n Saved the training verbose and training loss at {SAVE_CLS_DIR}\n")

if not SAVE_AFTER_EVERY_EPOCH:
    model.save_pretrained(MODEL_SAVE_PATH)
    print("Saved model to disk after final epoch !")

#%% Create a HF pipeline for easy inference.

pipe = pipeline("visual-question-answering", model=model, device=device,
                tokenizer=processor.tokenizer, image_processor=processor.image_processor)

pipe.save_pretrained(PIPELINE_SAVE_PATH)
print("Created HF pipeline and saved it as well.")
#%% Test inference on a sample
idx = 5
image_path = dic["image_path"][idx]

image = Image.open(image_path)
object_name = dic['categoryName'][idx]

print(f"object_name: {object_name}")
pred = pipe(image, object_name, top_k=1)
print(f"prection: {pred}")
print(f"Actual location: {dic['positionName'][idx]}")


#%%
from datetime import timedelta

get_hhmmss = lambda secs: str(timedelta(seconds=secs))

end_time = time.time()
total_time = end_time - start_time

# print(f"Start Time: {get_hhmmss(start_time)}")
# print(f"End time: {get_hhmmss(end_time)}")
print(f"Total time taken: {get_hhmmss(total_time)}")