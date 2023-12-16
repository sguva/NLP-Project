''' 
Code to generate training and validation data using COCO 2017 dataset 


'''
import json
from time import time
import csv
import os
import pandas as pd

DATASET_PATH = os.environ['DATASET_PATH']
SAVE_DIR = os.environ['SAVE_DIR']

SAVE_DATAGEN_DIR = os.path.join(SAVE_DIR, "data_generation")
os.makedirs(SAVE_DATAGEN_DIR, exist_ok=True)

for train_or_val in ["train", "val"]:
    print(f"___________Working on {train_or_val} data_________________")
    instances_json_path = os.path.join(DATASET_PATH, "annotations", f"instances_{train_or_val}2017.json")
    filter_data_with_position_csv_file = os.path.join(SAVE_DATAGEN_DIR, f"fitlered_data_with_position_{train_or_val}.csv")
    extracted_data_json = os.path.join(SAVE_DATAGEN_DIR, f"extracted_data_{train_or_val}.json")
    filtered_data_json = os.path.join(SAVE_DATAGEN_DIR, f"filtered_data_{train_or_val}.json")
    questions_and_answers_file = os.path.join(SAVE_DATAGEN_DIR, f"generated_questions_and_answers_{train_or_val}.csv")

    def extractData():
        start_time = time()
        data = None
        extracted_data = []
        try:
            with open(instances_json_path) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"{instances_json_path} File not found.")
            return

        if "annotations" in data:
            annotations_data = data["annotations"]
            for img_annotation in annotations_data:
                extracted_data.append(
                    {
                        "image_id": img_annotation["image_id"],
                        "bbox": img_annotation["bbox"],
                        "category_id": img_annotation["category_id"],
                    }
                )

        else:
            print("No 'annotations' key found in the JSON file.")

        if extracted_data:
            print(extracted_data[:10])
        else:
            print("Extracted data empty")
            return

        extracted_data = {"annotations": extracted_data}

        print("Time taken: ", (time() - start_time) / 60, " minutes.")

        with open(extracted_data_json, "w") as f:
            json.dump(extracted_data, f)

        print(f"Written {len(extracted_data['annotations'])} annotations to file")
        print("Done. Time taken: ", (time() - start_time) / 60, " minutes.")


    def filterToRemoveMultiples():
        # filter out annotations with multiple objects of same class in the image

        # Read data from the file
        start_time = time()
        file_path = extracted_data_json
        with open(file_path, "r") as file:
            data = json.load(file)

        # Create a dictionary to store unique instances of category_id for each image_id
        filtered_data = {}
        for annotation in data["annotations"]:
            image_id = annotation["image_id"]
            category_id = annotation["category_id"]
            bbox = annotation["bbox"]
            if (image_id, category_id) not in filtered_data:
                filtered_data[(image_id, category_id)] = {"bbox": bbox, "error": False}
            else:
                filtered_data[(image_id, category_id)]["error"] = True

        ans_list = []
        # dictionary without items with error = True in value
        for key, value in filtered_data.items():
            if not value["error"]:
                ans_list.append(
                    {"image_id": key[0], "bbox": value["bbox"], "category_id": key[1]}
                )

        ans_json = {"annotations": ans_list}
        # write to file
        with open(filtered_data_json, "w") as file:
            json.dump(ans_json, file)

        print(f"Written {len(ans_json['annotations'])} annotations to file")
        print("Done. Time taken: ", (time() - start_time) / 60, " minutes.")


    def getCategoryNames():
        # Read data from the file
        start_time = time()
        with open(instances_json_path, "r") as file:
            data = json.load(file)
        # Create a dictionary to store unique instances of category_id for each image_id
        categoryNames = {}
        for category in data["categories"]:
            categoryNames[category["id"]] = [category["name"], category["supercategory"]]
        print("Done. Time taken: ", (time() - start_time) / 60, " minutes.")
        return categoryNames


    def determinePositionAndMakeCsvFile():
        positionMap = {
            0: "topLeft",
            1: "topCenter",
            2: "topRight",
            3: "middleLeft",
            4: "middleCenter",
            5: "middleRight",
            6: "bottomLeft",
            7: "bottomCenter",
            8: "bottomRight",
        }
        start_time = time()
        dimensionsDictionary = {}
        data = None
        try:
            with open(instances_json_path) as f:
                data = json.load(f)
        except FileNotFoundError:
            print("File not found.")
            return

        if "images" in data:
            images_data = data["images"]
            for img in images_data:
                dimensionsDictionary[img["id"]] = {
                    "width": img["width"],
                    "height": img["height"],
                }
        else:
            print("No 'images' key found in the JSON file.")
            return
        # read data from filtered_data.json
        file_path = filtered_data_json
        with open(file_path, "r") as file:
            filteredData = json.load(file)
        # create a dictionary to store the final data
        finalData = []
        ct = 0
        categoryMap = getCategoryNames()
        for annotation in filteredData["annotations"]:
            image_id = annotation["image_id"]
            bbox = annotation["bbox"]
            category_id = annotation["category_id"]
            height = dimensionsDictionary[image_id]["height"]
            width = dimensionsDictionary[image_id]["width"]
            bboxCenter = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
            # if the image is divided into 3 parts vertically and 3 parts horizontally, which part is the bboxCenter in?
            # 0, 1, 2
            # 3, 4, 5
            # 6, 7, 8
            if bboxCenter[0] < width / 3:
                if bboxCenter[1] < height / 3:
                    position = 0
                elif bboxCenter[1] < 2 * height / 3:
                    position = 3
                else:
                    position = 6
            elif bboxCenter[0] < 2 * width / 3:
                if bboxCenter[1] < height / 3:
                    position = 1
                elif bboxCenter[1] < 2 * height / 3:
                    position = 4
                else:
                    position = 7
            else:
                if bboxCenter[1] < height / 3:
                    position = 2
                elif bboxCenter[1] < 2 * height / 3:
                    position = 5
                else:
                    position = 8
            categoryMapping = categoryMap[category_id]
            finalData.append(
                [
                    ct,
                    image_id,
                    bbox,
                    category_id,
                    height,
                    width,
                    position,
                    positionMap[position],
                    categoryMapping[0],
                    categoryMapping[1],
                ]
            )
            ct += 1
        # write to csv file with headers
        with open(filter_data_with_position_csv_file, "w") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "row_id",
                    "image_id",
                    "bbox",
                    "category_id",
                    "height",
                    "width",
                    "position",
                    "positionName",
                    "categoryName",
                    "superCategoryName",
                ]
            )
            writer.writerows(finalData)
        print("Done. Time taken: ", (time() - start_time) / 60, " minutes.")

    # Extract required data from COCO dataset and create data for classification task.
    extractData()
    filterToRemoveMultiples()
    determinePositionAndMakeCsvFile()

    # Using the extracted information, generate Question answers pairs for text generation task.
    df = pd.read_csv(filter_data_with_position_csv_file)

    image_dict = {}

    for index, row in df.iterrows():
        image_id = row['image_id']
        category_name = row['categoryName']
        position_name = row['positionName']
        
        if image_id not in image_dict:
            image_dict[image_id] = []
        
        image_dict[image_id].append((category_name, position_name))

    ct = 0
    for k, v in image_dict.items():
        ct += 1
        print(k, v)
        if ct == 4:
            break

    print(len(image_dict))

    def constructQuestionAndAnswer(obj1, pos1, variationNum):
        questionList = [
            f"Where is the {obj1} located in the image?",
            f"What is the position of the {obj1} in the image?",
            f"Can you tell me where the {obj1} is in the picture?",
            f"Describe the location of the {obj1} in the image.",
            f"Locate the {obj1} in the image.",
            f"Identify the position of the {obj1} in the picture.",
            f"Where can I find the {obj1} in the image?",
            f"Point out the location of the {obj1}.",
            f"Tell me about the placement of the {obj1} in the image.",
            f"Give me the whereabouts of the {obj1} in the picture.",
            f"Specify the position of the {obj1} in the image.",
            f"Elaborate on the whereabouts of the {obj1} in the picture.",
            f"In which part of the image is the {obj1} situated?",
            f"Pinpoint the location of the {obj1} in the image.",
            f"Describe where the {obj1} is placed in the picture.",
            f"Locate the the {obj1} within the image.",
            f"Indicate the position of the {obj1} in the picture.",
            f"Describe the spatial location of the {obj1} in the image.",
            f"Identify the location of the {obj1} in the picture.",
            f"Specify the exact position of the {obj1} in the image.",
            f"Tell me about the spatial position of the {obj1} in the picture.",
            f"What quadrant is the {obj1} in the image?",
            f"Which area of the image is the {obj1} located in?",
            f"Tell me the spot where the {obj1} is placed in the picture.",
            f"Provide information about the location of the {obj1} in the image.",
            f"Identify the general area where the {obj1} is located in the picture.",
            f"Elaborate on the general whereabouts of the {obj1} in the picture.",
            f"Tell me about the placement of the {obj1} in the image."
            ]

        
        variationNum = variationNum % len(questionList)
        answer = f"The {obj1} is located at the {pos1} of the image."
        return questionList[variationNum], answer

    def combine2QuestionsAndAnswers(q1, a1, q2, a2, obj1, obj2):
        combinations = []
        fixedAnswer = a1[:-14] + ' and ' + a2[0].lower() + a2[1:]
        combinations.append( (q1[:-1] + ' and ' + q2[0].lower() + q2[1:], fixedAnswer))
        q1 = q1.replace(obj1, obj1 + " and the " + obj2)
        q1 = q1.replace(" is ", " are ")
        combinations.append( (q1, fixedAnswer))
        return combinations

    questionsAndAnswers = []
    varNum = 0
    for k, v in image_dict.items():
        if len(v) == 1:
            obj1, pos1 = v[0]
            q, a = constructQuestionAndAnswer(obj1, pos1, varNum)
            questionsAndAnswers.append((k, q, a))
            varNum += 1
        else:
            for i in range(len(v)-1):
                obj1, pos1 = v[i]
                obj2, pos2 = v[i+1]
                variationsNeeded = 3
                for j in range(variationsNeeded):
                    q1, a1 = constructQuestionAndAnswer(obj1, pos1, varNum)
                    q2, a2 = constructQuestionAndAnswer(obj2, pos2, varNum%11 + varNum%7)
                    questionsAndAnswers.append((k, q1, a1))
                    questionsAndAnswers.append((k, q2, a2))
                    qAndAs = combine2QuestionsAndAnswers(q1, a1, q2, a2, obj1, obj2)
                    for q, a in qAndAs:
                        questionsAndAnswers.append((k, q, a))
                    varNum += 1
                
    print(f"{len(questionsAndAnswers)} questions and answers generated.")


    for i in range(len(questionsAndAnswers)):
        questionsAndAnswers[i] = (i,) + questionsAndAnswers[i]

    questionsAndAnswers[:10]

    filename = questions_and_answers_file
    delimiter = '|'

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=delimiter)
        writer.writerow(['id', 'image_id', 'question', 'answer'])
        writer.writerows(questionsAndAnswers)

    print(f"\nQuestions and answers saved to {filename}.\n")