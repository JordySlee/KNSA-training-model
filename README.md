# KNSA-training-model

# Authors

Jordy Slee \n
Steff Burgering
Samuel Kuik

# Pro-tip

Use a virtual environment if you want to keep your default python installation intact!

To create and activate a virtual environment:

* python -m venv .venv
* source .venv/bin/activate

## Mac/Linux-users

* pip install -r requirements.txt
* cd into: yolov5-repository/yolov5
* pip install -r requirements.txt

## Windows-users

pip install: 
* wheel
* matplotlib
* numpy
* pandas
* tensorflow
* pandas
* opencv-python

* cd into: yolov5-repository/yolov5
* pip install -r requirements.txt

## Training the model

To train the model:
* cd into 'yolov5-repository/yolov5'. 
* run: > python train.py --data KNSA-shooting-target-v10/data.yaml --weights runs/train/exp15/best.py --img 416

This trains the model with the default values. 
Flags can be used (e.g. '--epoch 200') to change these values.
To change the default values, open 'train.py' and scroll to line 477. All modifier values are found here.

## Running the trained model

To run the trained model with the sample images, run '> python yolov5.py'