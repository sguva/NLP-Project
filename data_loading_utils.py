#%%
import os
import csv
# from datasets import load_dataset, Dataset, DatasetDict
from PIL import Image
import json
import sys
from tqdm import tqdm
import torch

DATASET_PATH = os.environ['DATASET_PATH']
SAVE_DIR = os.environ['SAVE_DIR']

SAVE_DATAGEN_DIR = os.path.join(SAVE_DIR, "data_generation")
os.makedirs(SAVE_DATAGEN_DIR, exist_ok=True)


TRAIN_IMAGES_PATH = os.path.join(DATASET_PATH, "train2017")
VAL_IMAGES_PATH = os.path.join(DATASET_PATH, "val2017")

ANNOTATIONS_PATH = os.path.join(DATASET_PATH, "annotations")
TRAIN_INSTANCES_FILEPATH = os.path.join(ANNOTATIONS_PATH, "instances_train2017.json")
VAL_INSTANCES_FILEPATH = os.path.join(ANNOTATIONS_PATH, "instances_val2017.json")

# Check existence of directories
directories = [DATASET_PATH, TRAIN_IMAGES_PATH, VAL_IMAGES_PATH, ANNOTATIONS_PATH]

all_directories_exist = True
for directory in directories:
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        all_directories_exist = False

if all_directories_exist:
    print("All required directories exist of COCO dataset.")

SAVE_ENCODED_DATA_TO_DISK_DIR_CLS = os.path.join(SAVE_DATAGEN_DIR, f"saved_as_pth/classification")
os.makedirs(SAVE_ENCODED_DATA_TO_DISK_DIR_CLS, exist_ok=True)

SAVE_ENCODED_DATA_TO_DISK_DIR_AG = os.path.join(SAVE_DATAGEN_DIR, f"saved_as_pth/answer_generation")
os.makedirs(SAVE_ENCODED_DATA_TO_DISK_DIR_AG, exist_ok=True)

ids = list(range(0,9))
id2label = {0: "topLeft", 1: "topCenter", 2: "topRight", 3: "middleLeft", 4: "middleCenter", 5: "middleRight", 6: "bottomLeft", 7: "bottomCenter", 8: "bottomRight"}
labels = list(id2label.values())
label2id = {label: idx for idx, label in enumerate(labels)}

'''
For data preprossing and loading, we are processing the data manually instead of using the hugging face's datasets 
and datasets's map method. This is becuase, we want to load entire data into main memory so that it will save time 
over fetching from disk. But ofcourse, if we are trianing for huge number of samples then we need more RAM.
As Rivanna had GPU machines with lots of RAM, we did this way to save time over fetching data from disk after every iteration.
With HF's dataset there is an in_memory=True karg but it was slower when compared to what we are doing in this file.
Also, we save the encoded data with to disk to avoid repetition when loaded next time.
'''

#%% get_data_encoded_ga returns the encoded image and text data for trianing or validation for classification task.
# The text data is not encoded by this function but will be encoded by the collate_fn itself for batching concerns.

def get_data_encoded_ga(processor, train_or_val, num_files=-1, sanity_check=False):

    questions_and_answers_file = os.path.join(SAVE_DATAGEN_DIR, f"generated_questions_and_answers_{train_or_val}.csv")
    save_encoded_data_to_disk_file_name = os.path.join(SAVE_ENCODED_DATA_TO_DISK_DIR_AG, f"answer_generation_{train_or_val}_{num_files}.pth")

    if os.path.isfile(save_encoded_data_to_disk_file_name):
        data_encoded = torch.load(save_encoded_data_to_disk_file_name)
        print(f"Found saved data and loaded from disk which has length: {len(data_encoded)}")

    else:
        print(f"Loading {train_or_val} data dictionary............")
        dic = get_data_dic(questions_and_answers_file, train_or_val=train_or_val, num_files=num_files)
        print(f"Len of dic: {len(dic['question'])}")
        data_encoded = []
        encoded_images_dict = {}
        for item in tqdm(zip(dic['image_id'], dic['image_path'], dic['question'], dic['answer']), total=len(dic['question']), desc='Encoding image data'):
            image_id, image_path, question, answer = item

            if image_id not in encoded_images_dict:
                image = Image.open(image_path).convert('RGB').resize((224, 224))
                encoding = processor(images=image, padding="max_length", return_tensors="pt")
                # remove batch dimension
                encoding = {k: v.squeeze() for k, v in encoding.items()}
                encoded_images_dict[image_id] = encoding
            
            encoding = encoded_images_dict[image_id].copy()

            if train_or_val == "train":
                text = f"Question: {question} Answer: {answer}"
            else:
                text = f"Question: {question} Answer:"
            # print(text)
            encoding["text"] = text
            data_encoded.append(encoding)
        print(f"Saving data_encoded to disk.... at {save_encoded_data_to_disk_file_name}")
        torch.save(data_encoded, save_encoded_data_to_disk_file_name)
        print("Done saving!")

    if sanity_check:
        n = 25
        for i in range(n):
            print(data_encoded[i]['text'])
    
    if train_or_val == "val":
        dic = get_data_dic(questions_and_answers_file, train_or_val=train_or_val, num_files=num_files)
    else:
        # Short dictionary with 12 files from training data for inference at the end.
        dic = get_data_dic(questions_and_answers_file, train_or_val=train_or_val, num_files=12)

    return data_encoded, dic

#%% get_data_encoded_cls returns the encoded image and text data for trianing or validation for classification task.
# For classification we are tokenizing to max length = 8 as inputs are just object names and not sentences.
# Hence there wouldn't be any batching issue and can encode invidual samples as all of them will have fixed length of 8.

def get_data_encoded_cls(processor, train_or_val, num_files=-1):
    
    filter_data_with_position_csv_file = os.path.join(SAVE_DATAGEN_DIR, f"fitlered_data_with_position_{train_or_val}.csv")
    save_encoded_data_to_disk_file_name = os.path.join(SAVE_ENCODED_DATA_TO_DISK_DIR_CLS, f"classification_{train_or_val}_{num_files}.pth")

    if os.path.isfile(save_encoded_data_to_disk_file_name):
        data_encoded = torch.load(save_encoded_data_to_disk_file_name)
        print(f"Found saved data and loaded from disk which has length: {len(data_encoded)}")

    else:
        print(f"Loading {train_or_val} data dictionary............")
        dic = get_data_dic_cls(filter_data_with_position_csv_file, train_or_val=train_or_val, num_files=num_files)
        print(f"Len of dic: {len(dic['image_id'])}")
        data_encoded = []
        encoded_images_dict = {}

        for item in tqdm(zip(dic['row_id'], dic['image_id'], dic['image_path'], dic['positionName'], dic['categoryName']), total=len(dic['image_id']), desc='Encoding data'):
            row_id, image_id, image_path, positionName, categoryName = item

            if image_id not in encoded_images_dict:
                image = Image.open(image_path).convert('RGB').resize((384, 384))
                encoding = processor.image_processor(images=image, return_tensors='pt')
                encoding = {k: v.squeeze() for k, v in encoding.items()}
                encoded_images_dict[image_id] = encoding
            
            encoding = encoded_images_dict[image_id].copy()

            text = categoryName
            encoding_text = processor.tokenizer(text=text, padding="max_length", max_length=8, truncation=True, return_tensors='pt')
            encoding_text = {k: v.squeeze() for k, v in encoding_text.items()}

            # Append text encodings to image encodings.
            encoding.update(encoding_text)

            # Append one-hot encoding of labels
            target = torch.zeros(len(labels))
            target[label2id[positionName]] = 1
            encoding["labels"] = target

            data_encoded.append(encoding)
        
        print(f"Saving data_encoded to disk.... at {save_encoded_data_to_disk_file_name}")
        torch.save(data_encoded, save_encoded_data_to_disk_file_name)
        print("Done saving!")

    # Short dictionary with 12 files from training data for inference at the end.
    dic = get_data_dic_cls(filter_data_with_position_csv_file, train_or_val=train_or_val, num_files=12)

    return data_encoded, dic

#%% get_data_dic_image will directly load image instead of image_paths whereas get_data_dic loads image paths.
# Will return a dictionay of the format: {'row_id': [], 'image_id':[], 'image': [], 'question': [], 'answer': []}

def get_data_dic_image(csv_file_path, train_or_val, num_files=-1, save_or_load_file=False):

    save_dic_to_file_path = os.path.join(SAVE_DATAGEN_DIR, train_or_val + "_" + str(num_files) + "_images_dict.json")
    
    # If save_or_load_file = True and there's a dic that is saved to disk, directly load from it.
    if save_or_load_file and os.path.exists(save_dic_to_file_path):
        with open(save_dic_to_file_path, "r") as f:
            json_string = f.read()
            dic = json.loads(json_string)
        return dic

    img_id_fname = get_img_id_fname_dict(train_or_val)
    # print(f"Len of img_id_fname: {len(img_id_fname)}")
    
    dic = {'row_id': [], 'image_id':[], 'image': [], 'question': [], 'answer': []}

    images_dic = {}
    if train_or_val == "train":
        IMAGES_PATH = TRAIN_IMAGES_PATH
    else:
        IMAGES_PATH = VAL_IMAGES_PATH

    with open(csv_file_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader) # Skip the first row with column names
        n = 0

        for row in reader:
            row_id, image_id, question, answer = row[0].split('|')
            if image_id in img_id_fname:
                image_path = os.path.join(IMAGES_PATH, img_id_fname[image_id])

                if os.path.isfile(image_path):
                    dic["row_id"].append(row_id)
                    dic["image_id"].append(image_id)
                    if image_id not in images_dic:
                        images_dic[image_id] = Image.open(image_path).convert("RGB")
                    dic["image"].append(images_dic[image_id])
                    dic["question"].append(question)
                    dic["answer"].append(answer)
                    n += 1

                    if n == num_files:
                        break
            # else:
            #     print(f"Item missing: {image_id}")
    
    # If save_or_load_file = True, save the data the dic data to disk
    if save_or_load_file:
        json_string = json.dumps(dic)

        with open(save_dic_to_file_path, "w") as f:
            f.write(json_string)

    return dic


# returns dictionary of format: {'row_id': [], 'image_id':[], 'image_path': [], 'question': [], 'answer': []}

def get_data_dic(csv_file_path, train_or_val, num_files=-1, save_or_load_file = False):

    save_dic_to_file_path = os.path.join(SAVE_DATAGEN_DIR, train_or_val + "_" + str(num_files) + "_image_paths_dict.json")
    
    # If save_or_load_file = True and there's a dic that is saved to disk, directly load from it.
    if save_or_load_file and os.path.exists(save_dic_to_file_path):
        with open(save_dic_to_file_path, "r") as f:
            json_string = f.read()
            dic = json.loads(json_string)
        return dic

    img_id_fname = get_img_id_fname_dict(train_or_val)
    # print(f"Len of img_id_fname: {len(img_id_fname)}")
    
    dic = {'row_id': [], 'image_id':[], 'image_path': [], 'question': [], 'answer': []}

    if train_or_val == "train":
        IMAGES_PATH = TRAIN_IMAGES_PATH
    else:
        IMAGES_PATH = VAL_IMAGES_PATH

    with open(csv_file_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader) # Skip the first row with column names
        n = 0

        for row in reader:
            row_id, image_id, question, answer = row[0].split('|')
            if image_id in img_id_fname:
                image_path = os.path.join(IMAGES_PATH, img_id_fname[image_id])

                if os.path.isfile(image_path):
                    dic["row_id"].append(row_id)
                    dic["image_id"].append(image_id)
                    dic["image_path"].append(image_path)
                    dic["question"].append(question)
                    dic["answer"].append(answer)
                    n += 1

                    if n == num_files:
                        break
            # else:
            #     print(f"Item missing: {image_id}")
    
    # If save_or_load_file = True, save the data the dic data to disk
    if save_or_load_file:
        json_string = json.dumps(dic)

        with open(save_dic_to_file_path, "w") as f:
            f.write(json_string)

    return dic

#%% Same function as above but for classification task.

def get_data_dic_cls(csv_file_path, train_or_val, num_files=-1, save_or_load_file = False):

    save_dic_to_file_path = os.path.join(SAVE_DATAGEN_DIR, train_or_val + "_" + str(num_files) + "_dict_cls.json")
    
    # If save_or_load_file = True and there's a dic that is saved to disk, directly load from it.
    if save_or_load_file and os.path.exists(save_dic_to_file_path):
        with open(save_dic_to_file_path, "r") as f:
            json_string = f.read()
            dic = json.loads(json_string)
        return dic

    img_id_fname = get_img_id_fname_dict(train_or_val)
    # print(f"Len of img_id_fname: {len(img_id_fname)}")
    
    dic = {'row_id': [], 'image_id':[], 'image_path': [], 'positionName': [], 'categoryName': []}

    if train_or_val == "train":
        IMAGES_PATH = TRAIN_IMAGES_PATH
    else:
        IMAGES_PATH = VAL_IMAGES_PATH

    try:
        with open(csv_file_path, "r") as csv_file:
            reader = csv.reader(csv_file)
            next(reader) # Skip the first row with column names
            n = 0

            for row in reader:
    # ['row_id', 'image_id', 'bbox', 'category_id', 'height', 'width', 'position', 'positionName', 'categoryName', 'superCategoryName']
    # ['0', '289343', '[473.07, 395.93, 38.65, 28.67]', '18', '640', '529', '5', 'middleRight', 'dog', 'animal']
                row_id, image_id, positionName, categoryName = row[0], row[1], row[7], row[8]
                if image_id in img_id_fname:
                    image_path = os.path.join(IMAGES_PATH, img_id_fname[image_id])

                    if os.path.isfile(image_path):
                        dic["row_id"].append(row_id)
                        dic["image_id"].append(image_id)
                        dic["image_path"].append(image_path)
                        dic["positionName"].append(positionName)
                        dic["categoryName"].append(categoryName)
                        n += 1

                        if n == num_files:
                            break
                # else:
                #     print(f"Item missing: {image_id}")

    except FileNotFoundError:
        print("File not found.")
        return
    # If save_or_load_file = True, save the data the dic data to disk
    if save_or_load_file:
        json_string = json.dumps(dic)

        with open(save_dic_to_file_path, "w") as f:
            f.write(json_string)

    return dic

#%% Function to get the image_id and filename of COCO dataset images.

def get_img_id_fname_dict(train_or_val):

    save_to_file_path = os.path.join(SAVE_DATAGEN_DIR, train_or_val+"_img_id_fname.json")

    # If there's a dict that is saved to disk, directly load from it.
    if os.path.exists(save_to_file_path):
        with open(save_to_file_path, "r") as f:
            json_string = f.read()
            img_id_fname = json.loads(json_string)
        return img_id_fname
    
    img_id_fname = {}
    if train_or_val == "train":
        instances_json_path = TRAIN_INSTANCES_FILEPATH
    else:
        instances_json_path = VAL_INSTANCES_FILEPATH

    try:
        with open(instances_json_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print("File not found.")
        return
    
    if "images" in data:
        images_data = data["images"]
        for img_data in images_data:
            img_id_fname[str(img_data["id"])] = img_data["file_name"]
    
    # Save the dictionary to disk.
    json_string = json.dumps(img_id_fname)

    with open(save_to_file_path, "w") as f:
        f.write(json_string)

    return img_id_fname