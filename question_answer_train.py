# Adapted from https://github.com/huggingface/notebooks/blob/main/peft/Fine_tune_BLIP2_on_an_image_captioning_dataset_PEFT.ipynb
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

from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model

train_or_val = "train"

DATASET_PATH = os.environ['DATASET_PATH']
SAVE_DIR = os.environ['SAVE_DIR']

SAVE_AG_DIR = os.path.join(SAVE_DIR, "answer_generation")
os.makedirs(SAVE_AG_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(SAVE_AG_DIR, "BLIP2_LoRA")

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

NUM_DATALOADER_WORKERS = NUM_PROC

#%% Load the data
from data_loading_utils import get_data_encoded_ga

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

data_encoded, dic = get_data_encoded_ga(processor, train_or_val, num_files=NUM_FILES)

# Create PyTorch Dataset
from torch.utils.data import Dataset, DataLoader

class VAQDataset(Dataset):
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

train_dataset = VAQDataset(data_encoded)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn, 
                            pin_memory=True, num_workers=NUM_DATALOADER_WORKERS)

# %% Load model BLIP2 model
print("\n Loading the model..........")
model = Blip2ForConditionalGeneration.from_pretrained("ybelkada/blip2-opt-2.7b-fp16-sharded", device_map="auto", load_in_8bit=True)

# LoraConfig
config = LoraConfig(
    r=256,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="lora_only",
    target_modules=["q_proj", "k_proj"]
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

# Open a file to log the output
save_training_log_path = os.path.join(SAVE_AG_DIR, 'training_log.txt')

# Initialize a list to store the loss values
loss_values = []

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
model.train()

with open(save_training_log_path, 'a') as f:
    print(f"\nTraining for {NUM_EPOCHS} epochs.........")
    for epoch in range(NUM_EPOCHS):
        # Initialize the loss for the current epoch
        epoch_loss = []
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device, torch.float16)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=input_ids)
            
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
plt.show()
plt.savefig(os.path.join(SAVE_AG_DIR, "training_loss.png"))
plt.close()

#%% Save model
if not SAVE_AFTER_EVERY_EPOCH:
    model.save_pretrained(MODEL_SAVE_PATH)
    print("Saved model to disk after final epoch !!")

# %% Inference
model.eval()

idx = 5
print(f"\nInference check on sample idx:{idx}")

image = Image.open(dic["image_path"][idx]).convert('RGB')
image

prompt = f"Question: {dic['question'][idx]} Answer:"
print(prompt)

inputs = processor(image, text=prompt, return_tensors="pt").to(device)#, torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=50)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(f"Generated answer: {generated_text}")
print(f"Actual answer:    {dic['answer'][idx]}")

# %%
from datetime import timedelta

get_hhmmss = lambda secs: str(timedelta(seconds=secs))

end_time = time.time()
total_time = end_time - start_time

# print(f"Start Time: {get_hhmmss(start_time)}")
# print(f"End time: {get_hhmmss(end_time)}")
print(f"\nTotal time taken: {get_hhmmss(total_time)}")