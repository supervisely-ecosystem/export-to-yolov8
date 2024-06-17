import os

import supervisely as sly
from distutils.util import strtobool
from dotenv import load_dotenv

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))


# region constants
TRAIN_TAG_NAME = "train"
VAL_TAG_NAME = "val"
ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ABSOLUTE_PATH)
TEMP_DIR = os.path.join(PARENT_DIR, "temp")
# endregion
sly.fs.mkdir(TEMP_DIR, remove_content_if_exists=True)

sly.logger.debug(f"Absolute path: {ABSOLUTE_PATH}, parent dir: {PARENT_DIR}")
sly.logger.info(f"App starting... TEMP_DIR: {TEMP_DIR}")

task_type = os.environ.get("modal.state.taskType", "segmentation")
include_visibility = bool(strtobool(os.environ.get("modal.state.includeVisibility", "false")))

IS_SEGM_TASK = task_type == "segmentation"
IS_POSE_EST_TASK = task_type == "pose"
INCLUDE_VISIBILTY_FLAG = IS_POSE_EST_TASK and include_visibility