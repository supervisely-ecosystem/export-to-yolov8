<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/115161827/235631478-31056b4a-4945-4962-aef0-4bd5b7b73956.png"/>

# Export to YOLOv8 format

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Preparation">Preparation</a> •
  <a href="#How-to-Run">How to Run</a> •
  <a href="#How-to-Use">How to Use</a>
</p>
  
[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/export-to-yolov8)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/export-to-yolov8)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/export-to-yolov8.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/export-to-yolov8.png)](https://supervise.ly)

</div>

# Overview

Transform images project in Supervisely ([link to format](https://docs.supervise.ly/data-organization/00_ann_format_navi)) to [YOLO v8 segmentation format](https://docs.ultralytics.com/tasks/segment/#dataset-format) and prepares downloadable `tar` archive.


# Preparation

Supervisely project has to contain only classes with shape `Polygon` or/and `Bitmap`. If your project has classes with other shapes, labels with other types of shapes will be skipped. We recommend you to use `Convert Class Shape` app to convert class shapes.

- [Convert Class Shape](https://ecosystem.supervise.ly/apps/convert-class-shape) - app allows to convert labels to different class shapes.  
    
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/convert-class-shape" src="xxx" height="70px" margin-bottom="20px"/>

In addition, YOLOv8 format implies the presence of train/val datasets. Thus, to split images on training and validation datasets you should assign  corresponding tags (`train` or `val`) to images. If image doesn't have such tags, it will be treated as `train`. We recommend to use app `Assign train/val tags to images`. 

- [Assign train/val tags to images](https://ecosystem.supervise.ly/apps/tag-train-val-test) - app allows to assign train/val tags to images.  
    
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/tag-train-val-test" src="xxx" height="70px" margin-bottom="20px"/>
    
# How to Run 
1. Add app to your team from [Ecosystem](https://ecosystem.supervise.ly/apps/export-to-yolov8) if it is not there.

2. Open context menu of project -> `Download as` -> `Export to YOLO v8 format` 
<img src="xxx" />


# How to Use

App creates task in `workspace tasks` list. Once app is finished, you will see download link to resulting tar archive. 

<img src="xxx" />

Resulting archive is saved in : 

`Current Team` -> `Files` -> `/tmp/supervisely/export/Export to YOLOv8 format/<task_id>/<project_id>_<project_name>.tar`. 

For example our file path is the following: 

`/tmp/supervisely/export/Export to YOLOv8 format/32803/20600_Demo.tar`.

<img src="xxx" />

If there are no `train` or `val` tags in project, special warning is printed. You will see all warnings in task logs.

<img src="xxx" />


Here is the example of `data_config.yaml` that you will find in archive:


```yaml
names: [kiwi, lemon]            # class names
colors: [[255,1,1], [1,255,1]]  # class colors
nc: 2                           # number of classes
train: ../lemons/images/train   # path to train imgs
val: ../lemons/images/val       # path to val imgs
```
