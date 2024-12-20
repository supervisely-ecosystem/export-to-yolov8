import asyncio
import os

import supervisely as sly

import src.functions as f
import src.globals as g
import src.workflow as w

api = None


class MyExport(sly.app.Export):
    def process(self, context: sly.app.Export.Context):
        global api
        api = sly.Api.from_env()

        project = api.project.get_info_by_id(id=context.project_id)
        if context.dataset_id is not None:
            datasets = [api.dataset.get_info_by_id(context.dataset_id)]
        else:
            datasets = api.dataset.get_list(project.id, recursive=True)

        w.workflow_input(api, project.id)

        images_count = 0
        for ds in datasets:
            images_count += ds.images_count
        result_dir_name = f"{project.id}_{project.name}"
        result_dir = os.path.join(g.TEMP_DIR, result_dir_name)
        sly.fs.mkdir(result_dir)

        dir_names = f.prepare_trainval_dirs(result_dir)

        meta_json = api.project.get_meta(project.id)
        meta = sly.ProjectMeta.from_json(meta_json)
        class_names = [obj_class.name for obj_class in meta.obj_classes]
        class_colors = [obj_class.color for obj_class in meta.obj_classes]

        max_kpts_count = 0

        for obj_class in meta.obj_classes:
            if issubclass(obj_class.geometry_type, sly.GraphNodes):
                max_kpts_count = max(
                    max_kpts_count,
                    len(obj_class.geometry_config[obj_class.geometry_type.items_json_field]),
                )

        f.check_tagmetas(meta)

        all_paths = {}
        all_ids = {}
        all_anns = {}
        total_images_count = 0
        skipped = []

        progress = sly.tqdm_sly(desc="Transforming annotations", total=images_count)
        for ds in datasets:
            train_info, val_info = f.process_images(
                api,
                meta,
                ds,
                class_names,
                progress,
                dir_names,
                skipped,
                max_kpts_count,
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

        download_progress = sly.tqdm_sly(desc="Downloading images", total=total_images_count)
        img_ids = []
        img_paths = []
        for ds in datasets:
            img_ids.extend(all_ids[ds.id]["train_ids"])
            img_ids.extend(all_ids[ds.id]["val_ids"])
            img_paths.extend(all_paths[ds.id]["train_paths"])
            img_paths.extend(all_paths[ds.id]["val_paths"])

        coro = api.image.download_paths_async(img_ids, img_paths, progress_cb=download_progress)
        loop = sly.utils.get_or_create_event_loop()
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            future.result()
        else:
            loop.run_until_complete(coro)

        f.prepare_yaml(result_dir_name, result_dir, class_names, class_colors, max_kpts_count)

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
            sly.logger.warning(msg)

        return result_dir


@sly.handle_exceptions(has_ui=False)
def main():
    try:
        app = MyExport()
        app.run()
        w.workflow_output(api, app.output_file)
    finally:
        if not sly.is_development():
            sly.logger.info(f"Remove temp directory: {g.TEMP_DIR}")
            sly.fs.remove_dir(g.TEMP_DIR)


if __name__ == "__main__":
    sly.main_wrapper("main", main)
