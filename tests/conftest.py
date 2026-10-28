"""Offline test harness: runs src/main.py's MyExport.process against an in-memory fake of sly.Api.

No Supervisely server is contacted. Run from the repo root:
    python -m pytest
with the SDK version from dev_requirements.txt installed (plus pytest), on Python 3.14 like the image.
"""

import importlib
import os
import sys
from types import SimpleNamespace

import pytest
import supervisely as sly

APP_MODULES = ("src.globals", "src.functions", "src.workflow", "src.main")


def _ann_json(ann):
    """Annotations are sly.Annotation or raw JSON (for shapes an older SDK cannot build)."""
    return ann if isinstance(ann, dict) else ann.to_json()


class FakeApi:
    """Just the calls the app makes. `datasets`: [(id, name, parent_id)], `images`: {ds_id: [(id, name, ann)]}."""

    def __init__(self, meta, datasets, images, project_id=1, project_name="proj"):
        self.meta = meta
        self.project_id = project_id
        self.ds = {
            ds_id: sly.DatasetInfo(**{f: None for f in sly.DatasetInfo._fields})._replace(
                id=ds_id,
                name=name,
                parent_id=parent,
                items_count=len(images.get(ds_id, [])),
                images_count=len(images.get(ds_id, [])),
                project_id=project_id,
            )
            for ds_id, name, parent in datasets
        }
        self.images = images
        self.downloaded = []
        self.workflow = []
        self.project = SimpleNamespace(
            get_info_by_id=lambda id: SimpleNamespace(id=id, name=project_name),
            get_meta=lambda id: meta if isinstance(meta, dict) else meta.to_json(),
        )
        self.dataset = SimpleNamespace(
            get_list=self._get_list, get_info_by_id=lambda id: self.ds[id], get_nested=self._get_nested
        )
        self.image = SimpleNamespace(get_list=self._image_list, download_paths_async=self._download_paths_async)
        self.annotation = SimpleNamespace(download_bulk_async=self._download_bulk_async)
        self.app = SimpleNamespace(
            workflow=SimpleNamespace(
                add_input_project=lambda pid: self.workflow.append(("in", pid)),
                add_output_file=lambda fi, meta=None: self.workflow.append(("out", fi.id)),
            )
        )

    def _get_list(self, project_id, recursive=False):
        return [d for d in self.ds.values() if recursive or d.parent_id is None]

    def _get_nested(self, project_id, dataset_id):
        out, frontier = [], [dataset_id]
        while frontier:
            children = [d for d in self.ds.values() if d.parent_id in frontier]
            out.extend(children)
            frontier = [d.id for d in children]
        return out

    def _image_list(self, dataset_id):
        return [
            SimpleNamespace(id=i, name=n, height=_ann_json(ann)["size"]["height"], width=_ann_json(ann)["size"]["width"])
            for i, n, ann in self.images.get(dataset_id, [])
        ]

    async def _download_bulk_async(self, dataset_id, image_ids, progress_cb=None):
        by_id = {i: ann for i, _, ann in self.images[dataset_id]}
        return [SimpleNamespace(image_id=i, annotation=_ann_json(by_id[i])) for i in image_ids]

    async def _download_paths_async(self, ids, paths, progress_cb=None):
        for i, p in zip(ids, paths):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(f"image-{i}".encode())
            self.downloaded.append((i, p))
            if progress_cb is not None:
                progress_cb(1)


@pytest.fixture
def run_export(tmp_path, monkeypatch):
    """Import the app with the given modal state and run MyExport.process. Returns result_dir."""

    def _run(api, task_type="segmentation", include_visibility=True, dataset_id=None):
        monkeypatch.setenv("modal.state.taskType", task_type)
        monkeypatch.setenv("modal.state.includeVisibility", "true" if include_visibility else "false")
        for name in APP_MODULES:
            sys.modules.pop(name, None)
        main = importlib.import_module("src.main")
        g = importlib.import_module("src.globals")
        monkeypatch.setattr(g, "TEMP_DIR", str(tmp_path / "temp"))
        monkeypatch.setattr(sly.Api, "from_env", classmethod(lambda cls, *a, **k: api))
        context = sly.app.Export.Context(
            team_id=1, workspace_id=1, project_id=api.project_id, dataset_id=dataset_id
        )
        return main.MyExport().process(context)

    return _run


def read_labels(result_dir, split):
    """{label file name: [lines]} for one split."""
    labels_dir = os.path.join(result_dir, "labels", split)
    out = {}
    for name in sorted(os.listdir(labels_dir)):
        with open(os.path.join(labels_dir, name)) as f:
            out[name] = f.read().splitlines()
    return out


def list_images(result_dir, split):
    return sorted(os.listdir(os.path.join(result_dir, "images", split)))
