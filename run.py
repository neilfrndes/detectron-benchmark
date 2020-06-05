import zipfile
from timeit import default_timer as timer

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

import common


# Images loading and initializing
setup_logger()
NUM_IMAGES = 3
NUM_LOOPS = 5

archive = zipfile.ZipFile('/home/ubuntu/detectron-benchmark/coco.zip', 'r')

img_array = []
with zipfile.ZipFile("coco.zip", "r") as f:
    for name in f.namelist()[1:NUM_IMAGES+1]:
        data = f.read(name)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
        img = np.transpose(img, (2, 0 ,1))
        img_tensor = torch.from_numpy(img)
        img_array.append({'image': img_tensor})

# Model loading and initializing
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.DEVICE = 'cpu'

model = build_model(cfg)
DetectionCheckpointer(model).load('/home/ubuntu/detectron-benchmark/rcnn.pkl') 
model.train(False);

# Perform inference

num_rows = len(img_array)

print(common.get_header())
benchmark_times = []
individual_times = []
for _ in range(NUM_LOOPS):

    start_time = timer()
    model(img_array)
    end_time = timer()

    total_time = end_time - start_time
    benchmark_times.append(total_time)

    individual_time = total_time*(10e3)/num_rows  # miliseconds
    individual_times.append(individual_time)

stats = common.calculate_stats(individual_times)
print(common.format_stats(NUM_IMAGES, stats))