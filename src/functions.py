import asyncio
import os

import numpy as np
import supervisely as sly
import yaml

import src.globals as g


def prepare_trainval_dirs(result_dir):
    train_imgs_dir = os.path.join(result_dir, "images/train")
    train_labels_dir = os.path.join(result_dir, "labels/train")
    sly.fs.mkdir(train_imgs_dir)
    sly.fs.mkdir(train_labels_dir)

    val_images_dir = os.path.join(result_dir, "images/val")
    val_labels_dir = os.path.join(result_dir, "labels/val")
    sly.fs.mkdir(val_images_dir)
    sly.fs.mkdir(val_labels_dir)
    return train_labels_dir, train_imgs_dir, val_images_dir, val_labels_dir


def check_tagmetas(meta):
    if meta.get_tag_meta(g.TRAIN_TAG_NAME) is None:
        sly.logger.warning(
            "Tag {!r} not found in project meta. Images without special tags will be marked as train".format(
                g.TRAIN_TAG_NAME
            )
        )

    if meta.get_tag_meta(g.VAL_TAG_NAME) is None:
        sly.logger.warning(
            "Tag {!r} not found in project meta. Images without special tags will be marked as train".format(
                g.VAL_TAG_NAME
            )
        )


def transform_segm_label(class_names, img_size, label: sly.Label):
    class_number = class_names.index(label.obj_class.name)
    if type(label.geometry) not in [sly.Bitmap, sly.Polygon, sly.AlphaMask]:
        raise RuntimeError(f'Unsupported "{label.geometry.geometry_name()}" geometry.')

    if type(label.geometry) in [sly.Bitmap, sly.AlphaMask]:
        new_obj_class = sly.ObjClass(label.obj_class.name, sly.Polygon)
        labels = label.convert(new_obj_class)
        if len(labels) == 0:
            return None
        for i, label in enumerate(labels):
            if i == 0:
                points = label.geometry.exterior_np
            else:
                points = np.concatenate((points, label.geometry.exterior_np), axis=0)
    else:
        points = label.geometry.exterior_np

    points = np.flip(points, axis=1)
    xy = []
    for point in points:
        xy.append(round(point[0] / img_size[1], 6))
        xy.append(round(point[1] / img_size[0], 6))
    xy_str = " ".join([str(point) for point in xy])
    result = f"{class_number} {xy_str}"
    return result


def transform_keypoint_label(class_names, img_size, label: sly.Label, max_kpts_count):
    class_number = class_names.index(label.obj_class.name)
    if type(label.geometry) != sly.GraphNodes:
        raise RuntimeError(f'Unsupported "{label.geometry.geometry_name()}" geometry.')
    bbox = label.geometry.to_bbox()
    x, y, w, h = bbox.center.col, bbox.center.row, bbox.width, bbox.height
    x, y, w, h = x / img_size[1], y / img_size[0], w / img_size[1], h / img_size[0]
    x, y, w, h = round(x, 6), round(y, 6), round(w, 6), round(h, 6)

    line = [class_number, x, y, w, h]
    for node in label.geometry.nodes.values():
        node: sly.Node
        line.append(round(node.location.col / img_size[1], 6))
        line.append(round(node.location.row / img_size[0], 6))
        if g.INCLUDE_VISIBILTY_FLAG:
            line.append(2 if not node.disabled else 1)
    if len(label.geometry.nodes) < max_kpts_count:
        for _ in range(max_kpts_count - len(label.geometry.nodes)):
            line.extend([0, 0])
            if g.INCLUDE_VISIBILTY_FLAG:
                line.append(0)

    return " ".join([str(x) for x in line])


def process_images(
    api: sly.Api,
    project_meta,
    ds,
    class_names,
    progress,
    dir_names,
    skipped_classes,
    max_kpts_count,
):
    train_ids = []
    train_img_paths = []
    train_anns = []
    val_ids = []
    val_image_paths = []
    val_anns = []
    train_count = 0
    val_count = 0

    train_labels_dir, train_imgs_dir, val_images_dir, val_labels_dir = dir_names

    def _write_new_ann(path, content):
        with open(path, "a") as f1:
            f1.write("\n".join(content))

    images_infos = api.image.get_list(ds.id)

    img_ids = [image_info.id for image_info in images_infos]
    img_names = [f"{ds.name}_{image_info.name}" for image_info in images_infos]
    ann_infos = []

    coro = api.annotation.download_bulk_async(ds.id, img_ids)
    loop = sly.utils.get_or_create_event_loop()
    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        ann_infos = future.result()
    else:
        ann_infos = loop.run_until_complete(coro)

    for img_id, img_name, ann_info, img_info in zip(img_ids, img_names, ann_infos, images_infos):
        ann_json = ann_info.annotation
        img_info: sly.ImageInfo
        try:
            ann = sly.Annotation.from_json(ann_json, project_meta)
        except Exception as e:
            sly.logger.warning(
                f"Some problem with annotation for image {img_info.name} (ID: {img_info.id}). Skipped...: {repr(e)}"
            )
            ann = sly.Annotation(img_size=(img_info.height, img_info.width))

        yolov8_ann = []
        for label in ann.labels:
            try:
                if g.IS_SEGM_TASK:
                    yolov8_line = transform_segm_label(class_names, ann.img_size, label)
                else:
                    yolov8_line = transform_keypoint_label(
                        class_names, ann.img_size, label, max_kpts_count
                    )
                if yolov8_line is not None:
                    yolov8_ann.append(yolov8_line)
            except Exception as e:
                sly.logger.debug(f"Label skipped: {e}")
                skipped_classes.append(
                    (label.obj_class.name, label.geometry.geometry_name(), img_name)
                )

        image_processed = False

        unique_image_name = f"{ds.id}_{img_info.name}"

        if ann.img_tags.get(g.VAL_TAG_NAME) is not None:
            val_ids.append(img_id)
            ann_path = os.path.join(
                val_labels_dir, f"{sly.fs.get_file_name(unique_image_name)}.txt"
            )
            val_anns.append(ann_path)

            _write_new_ann(ann_path, yolov8_ann)
            img_path = os.path.join(val_images_dir, unique_image_name)
            val_image_paths.append(img_path)
            image_processed = True
            val_count += 1

        if not image_processed or ann.img_tags.get(g.TRAIN_TAG_NAME) is not None:
            train_ids.append(img_id)
            ann_path = os.path.join(
                train_labels_dir, f"{sly.fs.get_file_name(unique_image_name)}.txt"
            )
            train_anns.append(ann_path)

            _write_new_ann(ann_path, yolov8_ann)
            img_path = os.path.join(train_imgs_dir, unique_image_name)
            train_img_paths.append(img_path)
            image_processed = True
            train_count += 1

        progress.iter_done_report()

    sly.logger.info(
        f"DATASET '{ds.name}': {train_count} images for train, {val_count} images for validation"
    )

    train_info = [train_ids, train_img_paths, train_anns]
    val_info = [val_ids, val_image_paths, val_anns]
    return train_info, val_info


def prepare_yaml(result_dir_name, result_dir, class_names, class_colors, kpts_count):
    data_yaml = {
        "train": "../{}/images/train".format(result_dir_name),
        "val": "../{}/images/val".format(result_dir_name),
        "nc": len(class_names),
        "names": class_names,
        "colors": class_colors,
    }

    if g.IS_POSE_EST_TASK:
        data_yaml["kpt_shape"] = [kpts_count, 3 if g.INCLUDE_VISIBILTY_FLAG else 2]

    config_path = os.path.join(result_dir, "data_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=None)
