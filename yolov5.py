import torch
import cv2
import os, os.path


path_to_local_repo = 'yolov5-repository/yolov5'

# path=f'{path_to_local_repo}/runs/train/LATEST EXP/weights/best.pt'
model = torch.hub.load(path_to_local_repo, "custom", path=f'{path_to_local_repo}/runs/train/exp15/weights/best.pt', source='local')

# When using the complete dataset:
# Copy folder with complete dataset to the root of this repository (~/KNSA_Project/KNSA-training-model/{PLACE DATASET HERE})
# rename folder to 'complete-set'

# Read all files in folder
# file_list = os.listdir("complete-set")

# provide count if wanted
# file_count = len(file_list)
# print(file_count)

# make list of images
# images=[]

# read all images into open-cv2 and append to list
# for file in file_list:
#     img = cv2.imread(f"complete-set/{file}")
#     images.append(img)

# Comment out lines 34-39

model.max_det = 5

images = []

for i in range(5):
    i+=1
    img = cv2.imread(f"targets/sample{i}.jpg")[..., ::-1]
    images.append(img)

results = model(images, size=640)

results.print()
results.show()

print(results.pandas().xyxy)