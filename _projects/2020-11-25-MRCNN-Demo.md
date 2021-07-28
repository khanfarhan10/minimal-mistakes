---
title: "Mask RCNN"
excerpt: "Object Detection and Instance Segmentation using the State of the Art Mask-RCNN Neural Network."
toc: true
toc_sticky: true
toc_label: "Contents"

categories:
  - projects

tags:
  - Mask-RCNN
  - Object Detection
  - Instance Segmentation
  - Neural Networks
  - Deep Learning

last_modified_at: 2020-11-01T08:06:00-05:00
og_image: "/assets/images/Mask_RCNN_demo_files/Mask_RCNN_demo_9_1.png"
header:
  teaser: "/assets/images/Mask_RCNN_demo_files/Mask_RCNN_demo_9_1.png"
  overlay_image: "/assets/images/Mask_RCNN_demo_files/Mask_RCNN_demo_9_1.png"
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Image Source : Author"
  actions:
    - label: "GitHub Code by Facebook (FAIR)"
      url: "https://github.com/matterport/Mask_RCNN"
    - label: "Alternate Easy Implementation Code (Preferred)"
      url: "https://github.com/ayoolaolafenwa/PixelLib"

use_math: true
---

# Mask R-CNN Demo

A quick intro to using the pre-trained model to detect and segment objects.

```python
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

%matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
```

    Using TensorFlow backend.

## Configurations

We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the `CocoConfig` class in `coco.py`.

For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the `CocoConfig` class and override the attributes you need to change.

```python
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
```

    Configurations:
    BACKBONE_SHAPES                [[256 256]
     [128 128]
     [ 64  64]
     [ 32  32]
     [ 16  16]]
    BACKBONE_STRIDES               [4, 8, 16, 32, 64]
    BATCH_SIZE                     1
    BBOX_STD_DEV                   [ 0.1  0.1  0.2  0.2]
    DETECTION_MAX_INSTANCES        100
    DETECTION_MIN_CONFIDENCE       0.5
    DETECTION_NMS_THRESHOLD        0.3
    GPU_COUNT                      1
    IMAGES_PER_GPU                 1
    IMAGE_MAX_DIM                  1024
    IMAGE_MIN_DIM                  800
    IMAGE_PADDING                  True
    IMAGE_SHAPE                    [1024 1024    3]
    LEARNING_MOMENTUM              0.9
    LEARNING_RATE                  0.002
    MASK_POOL_SIZE                 14
    MASK_SHAPE                     [28, 28]
    MAX_GT_INSTANCES               100
    MEAN_PIXEL                     [ 123.7  116.8  103.9]
    MINI_MASK_SHAPE                (56, 56)
    NAME                           coco
    NUM_CLASSES                    81
    POOL_SIZE                      7
    POST_NMS_ROIS_INFERENCE        1000
    POST_NMS_ROIS_TRAINING         2000
    ROI_POSITIVE_RATIO             0.33
    RPN_ANCHOR_RATIOS              [0.5, 1, 2]
    RPN_ANCHOR_SCALES              (32, 64, 128, 256, 512)
    RPN_ANCHOR_STRIDE              2
    RPN_BBOX_STD_DEV               [ 0.1  0.1  0.2  0.2]
    RPN_TRAIN_ANCHORS_PER_IMAGE    256
    STEPS_PER_EPOCH                1000
    TRAIN_ROIS_PER_IMAGE           128
    USE_MINI_MASK                  True
    USE_RPN_ROIS                   True
    VALIDATION_STEPS               50
    WEIGHT_DECAY                   0.0001

## Create Model and Load Trained Weights

```python
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
```

## Class Names

The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.

To improve consistency, and to support training on data from multiple sources at the same time, our `Dataset` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our `Dataset` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.

To get the list of class names, you'd load the dataset and then use the `class_names` property like this.

```
# Load COCO dataset
dataset = coco.CocoDataset()
dataset.load_coco(COCO_DIR, "train")
dataset.prepare()

# Print class names
print(dataset.class_names)
```

We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

```python
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
```

## Run Object Detection

```python
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
```

    Processing 1 images
    image                    shape: (476, 640, 3)         min:    0.00000  max:  255.00000
    molded_images            shape: (1, 1024, 1024, 3)    min: -123.70000  max:  120.30000
    image_metas              shape: (1, 89)               min:    0.00000  max: 1024.00000

![Mask_RCNN_demo_9_1.png](/assets/images/Mask_RCNN_demo_files/Mask_RCNN_demo_9_1.png)

```python

```
