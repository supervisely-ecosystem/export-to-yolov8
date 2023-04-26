# export-to-yolov8
Export Supervisely project in YOLOv8 format (downloadable tar archive)

<!-- <div align="center" markdown>
<img src="https://user-images.githubusercontent.com/106374579/183683758-89476d80-de3f-424f-9bfa-f1562703a168.png"/> -->

# Export masks from Supervisely to YOLOv8 format


<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Preparation">Preparation</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a>
</p>

[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/export-to-yolov8)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/export-to-yolov8.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/export-to-yolov8.png)](https://supervise.ly)

</div>

## Overview

Transform images project in Supervisely ([link to format](https://docs.supervise.ly/data-organization/00_ann_format_navi)) to [YOLO v8 segmentation format](https://docs.ultralytics.com/tasks/segment/#dataset-format) and prepares downloadable `tar` archive.


## Preparation

Supervisely project has to contain only classes with shape `Polygon` or/and `Bitmap`. If your project has classes with other shapes, labels with other types of shapes will be skipped. We recommend you to use [`Convert Class Shape`](https://ecosystem.supervise.ly/apps/convert-class-shape) app to convert class shapes. 

In addition, YOLO v8 format implies the presence of train/val datasets. Thus, to split images on training and validation datasets you should assign  corresponding tags (`train` or `val`) to images. If image doesn't have such tags, it will be treated as `train`. We recommend to use app [`Assign train/val tags to images`](https://ecosystem.supervise.ly/apps/tag-train-val-test). 


## How To Run 
**Step 1**: Add app to your team from [Ecosystem](https://ecosystem.supervise.ly/apps/export-to-yolov8) if it is not there.

**Step 2**: Open context menu of project -> `Download as` -> `Export to YOLO v8 format` 

<!-- <img src="https://i.imgur.com/bOUC5WH.png" width="600px"/> -->


## How to use

App creates task in `workspace tasks` list. Once app is finished, you will see download link to resulting tar archive. 

<!-- <img src="https://i.imgur.com/kXnmshv.png"/> -->

Resulting archive is saved in : 

`Current Team` -> `Files` -> `/tmp/supervisely/export/Export to YOLOv8 format/<task_id>/<project_id>_<project_name>.tar`. 

For example our file path is the following: 

`/tmp/supervisely/export/Export to YOLOv8 format/32803/20600_Demo.tar`.

<!-- <img src="https://i.imgur.com/hGrAyY0.png"/> -->

If there are no `train` or `val` tags in project, special warning is printed. You will see all warnings in task logs.

<!-- <img src="https://i.imgur.com/O5tshZQ.png"/> -->


Here is the example of `data_config.yaml` that you will find in archive:


```yaml
names: [kiwi, lemon]            # class names
colors: [[255,1,1], [1,255,1]]  # class colors
nc: 2                           # number of classes
train: ../lemons/images/train   # path to train imgs
val: ../lemons/images/val       # path to val imgs
```