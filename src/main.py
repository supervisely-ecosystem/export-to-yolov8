import os
import supervisely as sly
import yaml

from dotenv import load_dotenv


if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))


batch_size = 10
STORAGE_DIR = sly.app.get_data_dir()
TRAIN_TAG_NAME = "train"
VAL_TAG_NAME = "val"


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
    if label.geometry.geometry_name() not in ["polygon", "bitmap", "rectangle"]:
        sly.logger.warn(f"Class shape: {label.geometry}")
        raise RuntimeError(f"Label has unsupported geometry: {label.geometry.geometry_name()}.")

    if label.geometry.geometry_name() in ["bitmap", "rectangle"]:
        new_obj_class = sly.ObjClass(label.obj_class.name, sly.Polygon)
        label = label.convert(new_obj_class)[0]

    coords = label.to_json()["points"]["exterior"]
    xy = []
    for coord in coords:
        xy.append(str(round(coord[0] / img_size[1], 6)))
        xy.append(str(round(coord[1] / img_size[0], 6)))
    polygon_points = f"{class_number} {' '.join(xy)}"
    return polygon_points


def process_images(api, project_meta, ds, class_names, progress, dir_names):
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

        for img_id, img_name, ann_info in zip(img_ids, img_names, ann_infos):
            ann_json = ann_info.annotation
            ann = sly.Annotation.from_json(ann_json, project_meta)

            yolov5_ann = []
            for label in ann.labels:
                yolov5_ann.append(transform_label(class_names, ann.img_size, label))

            image_processed = False

            if ann.img_tags.get(VAL_TAG_NAME) is not None:
                val_ids.append(img_id)
                ann_path = os.path.join(val_labels_dir, f"{sly.fs.get_file_name(img_name)}.txt")
                val_anns.append(ann_path)

                _write_new_ann(ann_path, yolov5_ann)
                img_path = os.path.join(val_images_dir, img_name)
                val_image_paths.append(img_path)
                image_processed = True
                val_count += 1

            if not image_processed or ann.img_tags.get(TRAIN_TAG_NAME) is not None:
                train_ids.append(img_id)
                ann_path = os.path.join(train_labels_dir, f"{sly.fs.get_file_name(img_name)}.txt")
                train_anns.append(ann_path)

                _write_new_ann(ann_path, yolov5_ann)
                img_path = os.path.join(train_imgs_dir, img_name)
                train_img_paths.append(img_path)
                image_processed = True
                train_count += 1

        progress.iters_done_report(len(batch))

    sly.logger.info("Number of images in train == {}".format(train_count))
    sly.logger.info("Number of images in val == {}".format(val_count))

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
        data = yaml.dump(data_yaml, f, default_flow_style=None)


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
        result_dir = os.path.join(STORAGE_DIR, result_dir_name)
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

        progress = sly.Progress("Transformation ...", images_count)
        for ds in datasets:
            train_info, val_info = process_images(api, meta, ds, class_names, progress, dir_names)
            train_ids, train_img_paths, train_anns = train_info
            val_ids, val_image_paths, val_anns = val_info

            all_paths[ds.id] = {"train_paths": train_img_paths}
            all_paths[ds.id]["val_paths"] = val_image_paths

            all_ids[ds.id] = {"train_ids": train_ids}
            all_ids[ds.id]["val_ids"] = val_ids

            all_anns[ds.id] = {"train_anns": train_anns}
            all_anns[ds.id]["val_anns"] = val_anns

        images_count = len(all_paths[ds.id]["train_paths"]) + len(all_paths[ds.id]["val_paths"])
        download_progress = sly.Progress("Downloading images ...", images_count)
        for ds in datasets:
            api.image.download_paths(
                ds.id,
                all_ids[ds.id]["train_ids"],
                all_paths[ds.id]["train_paths"],
                download_progress.iters_done_report,
            )
            api.image.download_paths(
                ds.id,
                all_ids[ds.id]["val_ids"],
                all_paths[ds.id]["val_paths"],
                download_progress.iters_done_report,
            )

        prepare_yaml(result_dir_name, result_dir, class_names, class_colors)

        return result_dir


app = MyExport()
app.run()
