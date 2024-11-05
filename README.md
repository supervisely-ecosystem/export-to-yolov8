<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/115161827/235631478-31056b4a-4945-4962-aef0-4bd5b7b73956.png"/>

# Export to YOLOv8 format for instance segmentation and pose estimation tasks

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Preparation">Preparation</a> •
  <a href="#How-to-Run">How to Run</a> •
  <a href="#How-to-Use">How to Use</a> •
  <a href="#Related-apps">Related apps</a>
</p>
  
[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/export-to-yolov8)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/export-to-yolov8)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/export-to-yolov8.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/export-to-yolov8.png)](https://supervisely.com)

</div>

# Overview

This application is designed specifically for instance segmentation and pose estimation tasks.

It transforms datasets from the [Supervisely format](https://docs.supervisely.com/data-organization/00_ann_format_navi) to the YOLOv8 **segmentation format** or **pose estimation format**. 
By using this application, you can effortlessly convert your dataset and download as `tar` archive.

**Changelog**:

🏃 
`v1.0.0` -  Starting from this version application supports exporting keypoints (`graph` geometry) to YOLOv8 format for pose estimation tasks.

# Labels Format

The YOLOv8 format is a text-based format that is used to represent object detection, instance segmentation, and pose estimation datasets.

- Each image in the dataset has a corresponding text file with the same name as the image file and the `.txt` extension in the `labels` folder. 
- Each object is represented by a separate line in the file, containing the `class-index` and the coordinates of the bounding mask, normalized to the range of 0 to 1 (relative to the image dimensions). 

**Segmentation**:

The format for a single row in the **segmentation** dataset output files is as follows:

```
<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

In this format, `<class-index>` is the index of the class for the object, and `<x1> <y1> <x2> <y2> ... <xn> <yn>` are the bounding coordinates of the object's segmentation mask. The coordinates are separated by spaces.

Here is an example of the YOLO instance segmentation dataset format for a single image with two object instances:

```
0 0.6812 0.48541 0.67 0.4875 0.67656 0.487 0.675 0.489 0.66
1 0.5046 0.0 0.5015 0.004 0.4984 0.00416 0.4937 0.010 0.492 0.0104
```

Learn more about Yolov8 segmentation format [here](https://docs.ultralytics.com/datasets/segment/).

**Detection**:

- To convert Supervisely project to YOLOv8 format for object detection tasks, you can use the [Convert Supervisely to YOLO v5 format ](https://ecosystem.supervisely.com/apps/convert-supervisely-to-yolov5-format) app.
    
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/convert-supervisely-to-yolov5-format" src="https://github.com/supervisely-ecosystem/export-to-yolov8/releases/download/untagged-6331ae231a8468233aab/yolo_det.jpg" height="70px" margin-bottom="20px"/>

Learn more about Yolov8 segmentation format [here](https://docs.ultralytics.com/datasets/detect/).

**Pose**:

The format for a single row in the **pose estimation** dataset output files is as follows:


Format with Dim = 2

```txt
<class-index> <x> <y> <width> <height> <px1> <py1> <px2> <py2> ... <pxn> <pyn>
```

Format with Dim = 3

```txt
<class-index> <x> <y> <width> <height> <px1> <py1> <p1-visibility> <px2> <py2> <p2-visibility> <pxn> <pyn> <p2-visibility>
```

In this format, <class-index> is the index of the class for the object,<x> <y> <width> <height> are coordinates of bounding box, and <px1> <py1> <px2> <py2> ... <pxn> <pyn> are the pixel coordinates of the keypoints. The coordinates are separated by spaces.

Learn more about Yolov8 pose estimation format [here](https://docs.ultralytics.com/datasets/pose/).

# Preparation

Supervisely project has to contain only classes with shape `Polygon`, `Bitmap`, `AlphaMask` for segmentation tasks or `Graph` for pose estimation tasks. 
If your project has classes with other shapes, labels with other types of shapes will be skipped. We recommend you to use `Convert Class Shape` app to convert class shapes (supported for segmentation tasks only).

- The [Convert Class Shape](https://ecosystem.supervisely.com/apps/convert-class-shape) app allows to convert labels to different class shapes.  
    
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/convert-class-shape" src="https://user-images.githubusercontent.com/115161827/235643553-d5dd001e-22ef-4e74-a303-b7cfd251b7fd.png" height="70px" margin-bottom="20px"/>

In addition, YOLOv8 format implies the presence of train/val datasets. Thus, to split images on training and validation datasets you should assign  corresponding tags (`train` or `val`) to images. If image doesn't have such tags, it will be treated as `train`. We recommend to use app `Assign train/val tags to images`. 

- [Assign train/val tags to images](https://ecosystem.supervisely.com/apps/tag-train-val-test) - app allows to assign train/val tags to images.  
    
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/tag-train-val-test" src="https://user-images.githubusercontent.com/115161827/235643549-d0f4ea23-c75e-46f2-8767-3d786eb79207.png" height="70px" margin-bottom="20px"/>
    
# How to Run 
1. Add app to your team from [Ecosystem](https://ecosystem.supervisely.com/apps/export-to-yolov8) if it is not there.

2. Open context menu of project -> `Download as` -> `Export to YOLO v8 format` 
<img src="https://user-images.githubusercontent.com/115161827/235641219-43f67765-99ff-4ece-803b-3cbbb07011c4.png" />

You can also run the application from the Ecosystem
<img src="https://user-images.githubusercontent.com/115161827/235641214-50e93901-3c4b-4976-911b-c50940e84972.png" />

# How to Use

App creates task in `workspace tasks` list. Once app is finished, you will see download link to resulting tar archive. 

<img src="https://user-images.githubusercontent.com/115161827/235643943-8e4d6be2-56aa-46bf-b4bb-c017e93b32a0.png" />

Resulting archive is saved in : 

`Current Team` -> `Files` -> `/tmp/supervisely/export/Export to YOLOv8 format/<task_id>/<project_id>_<project_name>.tar`. 

For example our file path is the following: 

`/tmp/supervisely/export/Export to YOLOv8 format/32803/20600_Demo.tar`.

If there are no `train` or `val` tags in project, special warning is printed. You will see all warnings in task logs.

<img src="https://user-images.githubusercontent.com/115161827/235644472-16b3076e-7929-42c3-9f8c-7c1dcb0ca6be.png" />


Here is the example of `data_config.yaml` that you will find in archive:


```yaml
names: [kiwi, lemon]            # class names
colors: [[255,1,1], [1,255,1]]  # class colors
nc: 2                           # number of classes
train: ../lemons/images/train   # path to train imgs
val: ../lemons/images/val       # path to val imgs

# Keypoints (for pose estimation tasks)
kpt_shape: [17, 3]  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
```

# Related apps

- [Train YOLOv8](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/yolov8/train) - app allows to create custom YOLOv8 weights through training process.
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/yolov8/train" src="https://github.com/supervisely-ecosystem/yolov8/assets/115161827/82348f9a-38fc-4736-885c-d6786e37a218" height="70px" margin-bottom="20px"/>

- [Serve YOLOv8](https://ecosystem.supervisely.com/apps/yolov8/serve) - app allows to deploy YOLOv8 model as REST API service.   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/yolov8/serve" src="https://github.com/supervisely-ecosystem/yolov8/assets/115161827/721f5344-013c-4466-bc05-88cc3efef5ca" height="70px" margin-bottom="20px"/>
