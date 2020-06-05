#! /bin/bash

# This script is specific to the AWS deep learning AMI
conda activate pytorch_latest_p36

# Install the detectron2 package
pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html

# Download the coco dataset
wget http://images.cocodataset.org/zips/val2017.zip -O coco.zip
wget https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl -O rcnn.pkl

# Run the benchmark
python run.py