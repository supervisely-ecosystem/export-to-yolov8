import os
import supervisely as sly
import yaml
import time
import traceback
import numpy as np

from dotenv import load_dotenv

from supervisely.io.exception_handlers import handle_exception

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))


TRAIN_TAG_NAME = "train"
VAL_TAG_NAME = "val"

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ABSOLUTE_PATH)
sly.logger.debug(f"Absolute path: {ABSOLUTE_PATH}, parent dir: {PARENT_DIR}")

TEMP_DIR = os.path.join(PARENT_DIR, "temp")
sly.fs.mkdir(TEMP_DIR, remove_content_if_exists=True)
sly.logger.info(f"App starting... TEMP_DIR: {TEMP_DIR}")


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
    if meta.get_tag_meta(TRAIN_TAG_NAME) is None:
        sly.logger.warn(
            "Tag {!r} not found in project meta. Images without special tags will be marked as train".format(
                TRAIN_TAG_NAME
            )
        )

    if meta.get_tag_meta(VAL_TAG_NAME) is None:
        sly.logger.warn(
            "Tag {!r} not found in project meta. Images without special tags will be marked as train".format(
                VAL_TAG_NAME
            )
        )


def transform_label(class_names, img_size, label: sly.Label):
    class_number = class_names.index(label.obj_class.name)
    if type(label.geometry) not in [sly.Bitmap, sly.Polygon]:
        raise RuntimeError(f'Unsupported "{label.geometry.geometry_name()}" geometry.')

    if type(label.geometry) is sly.Bitmap:
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


def process_images(api, project_meta, ds, class_names, progress, dir_names, skipped_classes):
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
    for batch in sly.batched(images_infos):
        img_ids = [image_info.id for image_info in batch]
        img_names = [f"{ds.name}_{image_info.name}" for image_info in batch]
        ann_infos = api.annotation.download_batch(ds.id, img_ids)

        for img_id, img_name, ann_info, img_info in zip(img_ids, img_names, ann_infos, batch):
            ann_json = ann_info.annotation
            img_info: sly.ImageInfo
            try:
                ann = sly.Annotation.from_json(ann_json, project_meta)
            except Exception as e:
                sly.logger.warn(
                    f"Some problem with annotation for image {img_info.name} (ID: {img_info.id}). Skipped...: {repr(e)}"
                )
                ann = sly.Annotation(img_size=(img_info.height, img_info.width))

            yolov8_ann = []
            for label in ann.labels:
                try:
                    yolov8_line = transform_label(class_names, ann.img_size, label)
                    if yolov8_line is not None:
                        yolov8_ann.append(yolov8_line)
                except Exception as e:
                    sly.logger.info(f"Label skipped: {e}")
                    skipped_classes.append(
                        (label.obj_class.name, label.geometry.geometry_name(), img_name)
                    )

            image_processed = False

            if ann.img_tags.get(VAL_TAG_NAME) is not None:
                val_ids.append(img_id)
                ann_path = os.path.join(val_labels_dir, f"{sly.fs.get_file_name(img_name)}.txt")
                val_anns.append(ann_path)

                _write_new_ann(ann_path, yolov8_ann)
                img_path = os.path.join(val_images_dir, img_name)
                val_image_paths.append(img_path)
                image_processed = True
                val_count += 1

            if not image_processed or ann.img_tags.get(TRAIN_TAG_NAME) is not None:
                train_ids.append(img_id)
                ann_path = os.path.join(train_labels_dir, f"{sly.fs.get_file_name(img_name)}.txt")
                train_anns.append(ann_path)

                _write_new_ann(ann_path, yolov8_ann)
                img_path = os.path.join(train_imgs_dir, img_name)
                train_img_paths.append(img_path)
                image_processed = True
                train_count += 1

        progress.iters_done_report(len(batch))

    sly.logger.info(
        f"DATASET '{ds.name}': {train_count} images for train, {val_count} images for validation"
    )

    train_info = [train_ids, train_img_paths, train_anns]
    val_info = [val_ids, val_image_paths, val_anns]
    return train_info, val_info


def prepare_yaml(result_dir_name, result_dir, class_names, class_colors):
    data_yaml = {
        "train": "../{}/images/train".format(result_dir_name),
        "val": "../{}/images/val".format(result_dir_name),
        "nc": len(class_names),
        "names": class_names,
        "colors": class_colors,
    }

    config_path = os.path.join(result_dir, "data_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=None)


def download_batch_with_retry(api: sly.Api, dataset_id, image_ids, paths_to_save):
    retry_cnt = 5
    curr_retry = 0
    while curr_retry <= retry_cnt:
        try:
            image_nps = api.image.download_nps(dataset_id, image_ids)
            if len(image_nps) != len(image_ids):
                raise RuntimeError(
                    f"Downloaded {len(image_nps)} images, but {len(image_ids)} expected."
                )
            for image_np, path in zip(image_nps, paths_to_save):
                sly.image.write(path, image_np)
            return
        except Exception as e:
            curr_retry += 1
            if curr_retry <= retry_cnt:
                time.sleep(2**curr_retry)
                sly.logger.warn(
                    f"Failed to download images, retry {curr_retry} of {retry_cnt}... Error: {e}"
                )
    raise RuntimeError(
        f"Failed to download images with ids {image_ids}. Check your data and try again later."
    )


class MyExport(sly.app.Export):
    def process(self, context: sly.app.Export.Context):
        api = sly.Api.from_env()

        project = api.project.get_info_by_id(id=context.project_id)
        if context.dataset_id is not None:
            datasets = [api.dataset.get_info_by_id(context.dataset_id)]
        else:
            datasets = api.dataset.get_list(project.id)

        images_count = 0
        for ds in datasets:
            images_count += ds.images_count
        result_dir_name = f"{project.id}_{project.name}"
        result_dir = os.path.join(TEMP_DIR, result_dir_name)
        sly.fs.mkdir(result_dir)

        dir_names = prepare_trainval_dirs(result_dir)

        meta_json = api.project.get_meta(project.id)
        meta = sly.ProjectMeta.from_json(meta_json)
        class_names = [obj_class.name for obj_class in meta.obj_classes]
        class_colors = [obj_class.color for obj_class in meta.obj_classes]

        check_tagmetas(meta)

        all_paths = {}
        all_ids = {}
        all_anns = {}
        total_images_count = 0
        skipped = []

        progress = sly.Progress("Transformation ...", images_count)
        for ds in datasets:
            train_info, val_info = process_images(
                api, meta, ds, class_names, progress, dir_names, skipped
            )
            train_ids, train_img_paths, train_anns = train_info
            val_ids, val_image_paths, val_anns = val_info

            all_paths[ds.id] = {"train_paths": train_img_paths}
            all_paths[ds.id]["val_paths"] = val_image_paths

            all_ids[ds.id] = {"train_ids": train_ids}
            all_ids[ds.id]["val_ids"] = val_ids

            all_anns[ds.id] = {"train_anns": train_anns}
            all_anns[ds.id]["val_anns"] = val_anns

            total_images_count += len(train_img_paths)
            total_images_count += len(val_image_paths)

        download_progress = sly.Progress("Downloading images ...", total_images_count)
        for ds in datasets:
            for train_batch in sly.batched(
                list(zip(all_ids[ds.id]["train_ids"], all_paths[ds.id]["train_paths"]))
            ):
                img_ids, img_paths = zip(*train_batch)
                download_batch_with_retry(api, ds.id, img_ids, img_paths)
                download_progress.iters_done_report(len(train_batch))
            for val_batch in sly.batched(
                list(zip(all_ids[ds.id]["val_ids"], all_paths[ds.id]["val_paths"]))
            ):
                img_ids, img_paths = zip(*val_batch)
                download_batch_with_retry(api, ds.id, img_ids, img_paths)
                download_progress.iters_done_report(len(val_batch))

        prepare_yaml(result_dir_name, result_dir, class_names, class_colors)

        sly.logger.info(f"Export finished. Total images count: {total_images_count}")
        skipped_cnt = len(skipped)
        if skipped_cnt > 0:
            _, _, skipped_images = zip(*skipped)
            skipped_images_cnt = len(set(skipped_images))
            cls_geom_pairs = {c: g for c, g, _ in skipped}

            msg = (
                f"{skipped_cnt} labels skipped on {skipped_images_cnt} images. "
                f"{len(cls_geom_pairs)} classes have unsupported geometry: {cls_geom_pairs}. "
                f"Use 'Convert Class Shape' app to convert unsupported shapes."
            )
            sly.logger.warn(msg)

        return result_dir


def main():
    app = MyExport()
    try:
        app.run()
        raise RuntimeError("Export finished successfully. But it should fail.")
    except Exception as e:
        exception_handler = handle_exception(e)
        if exception_handler:
            exception_handler.log_error_for_agent("main")
        else:
            sly.logger.error(
                traceback.format_exc(),
                exc_info=True,
                extra={
                    "main_name": "main",
                    "exc_str": repr(e),
                    "event_type": sly.EventType.TASK_CRASHED,
                },
            )
    finally:
        if not sly.is_development():
            sly.logger.info(f"Remove temp directory: {TEMP_DIR}")
            sly.fs.remove_dir(TEMP_DIR)


if __name__ == "__main__":
    main()
