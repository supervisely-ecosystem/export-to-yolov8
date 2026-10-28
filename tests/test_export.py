import os

import yaml
from conftest import FakeApi, list_images, read_labels
from helpers import META, PERSON_NODES, ann, mask, person, poly, rect


def test_segmentation_export(run_export):
    images = {
        10: [
            (1, "a.jpg", ann(poly([(10, 20), (10, 60), (50, 60)]), rect(0, 0, 5, 5))),
            (2, "b.jpg", ann(tags=["val"])),
        ]
    }
    api = FakeApi(META, [(10, "ds", None)], images)
    result_dir = run_export(api)
    assert list_images(result_dir, "train") == ["10_a.jpg"]
    assert list_images(result_dir, "val") == ["10_b.jpg"]
    # polygon: class 0, x y pairs; the rectangle is skipped in segmentation mode
    assert read_labels(result_dir, "train") == {"10_a.txt": ["0 0.1 0.1 0.3 0.1 0.3 0.5"]}
    assert read_labels(result_dir, "val") == {"10_b.txt": []}
    with open(os.path.join(result_dir, "data_config.yaml")) as f:
        cfg = yaml.safe_load(f)
    assert cfg["names"] == ["poly", "mask", "box", "person"]
    assert "kpt_shape" not in cfg
    assert api.workflow == [("in", 1)]


def test_bitmap_becomes_polygon(run_export):
    images = {10: [(1, "a.jpg", ann(mask()))]}
    result_dir = run_export(FakeApi(META, [(10, "ds", None)], images))
    (line,) = read_labels(result_dir, "train")["10_a.txt"]
    values = line.split()
    assert values[0] == "1"
    assert len(values) >= 7 and all(0 <= float(v) <= 1 for v in values[1:])


def test_pose_export(run_export):
    nose, left, right = PERSON_NODES
    p = person([(nose, 10, 20, False), (left, 30, 40, True), (right, 50, 60, False)])
    images = {10: [(1, "a.jpg", ann(p, poly([(1, 1), (1, 5), (5, 5)])))]}
    result_dir = run_export(FakeApi(META, [(10, "ds", None)], images), task_type="pose")
    (line,) = read_labels(result_dir, "train")["10_a.txt"]
    v = line.split()
    assert v[0] == "3"
    assert v[5:] == ["0.1", "0.1", "2", "0.2", "0.3", "1", "0.3", "0.5", "2"]
    with open(os.path.join(result_dir, "data_config.yaml")) as f:
        assert yaml.safe_load(f)["kpt_shape"] == [3, 3]


def test_pose_without_visibility(run_export):
    nose, left, right = PERSON_NODES
    p = person([(nose, 10, 20, False), (left, 30, 40, True), (right, 50, 60, False)])
    images = {10: [(1, "a.jpg", ann(p))]}
    result_dir = run_export(
        FakeApi(META, [(10, "ds", None)], images), task_type="pose", include_visibility=False
    )
    (line,) = read_labels(result_dir, "train")["10_a.txt"]
    assert line.split()[5:] == ["0.1", "0.1", "0.2", "0.3", "0.3", "0.5"]


def test_whole_project_with_nested_datasets(run_export):
    images = {10: [(1, "a.jpg", ann())], 11: [(2, "a.jpg", ann())], 12: [(3, "a.jpg", ann())]}
    datasets = [(10, "parent", None), (11, "child", 10), (12, "other", None)]
    result_dir = run_export(FakeApi(META, datasets, images))
    assert list_images(result_dir, "train") == ["10_a.jpg", "11_a.jpg", "12_a.jpg"]


def test_train_and_val_tags(run_export):
    images = {
        10: [
            (1, "t.jpg", ann(tags=["train"])),
            (2, "v.jpg", ann(tags=["val"])),
            (3, "both.jpg", ann(tags=["train", "val"])),
            (4, "none.jpg", ann()),
        ]
    }
    result_dir = run_export(FakeApi(META, [(10, "ds", None)], images))
    # an image tagged both train and val goes to both splits (unchanged behaviour)
    assert list_images(result_dir, "train") == ["10_both.jpg", "10_none.jpg", "10_t.jpg"]
    assert list_images(result_dir, "val") == ["10_both.jpg", "10_v.jpg"]
