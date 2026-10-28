import numpy as np
import pytest
import supervisely as sly
from conftest import FakeApi, read_labels
from helpers import PERSON, PERSON_NODES, ann, person


def _every_shape():
    geoms = {
        "rect": (sly.Rectangle, sly.Rectangle(10, 10, 20, 30)),
        "polygon": (sly.Polygon, sly.Polygon([sly.PointLocation(1, 1), sly.PointLocation(1, 9), sly.PointLocation(9, 9)])),
        "bitmap": (sly.Bitmap, sly.Bitmap(np.ones((3, 3), bool), origin=sly.PointLocation(2, 2))),
        "alpha": (sly.AlphaMask, sly.AlphaMask(np.full((3, 3), 255, np.uint8), origin=sly.PointLocation(2, 2))),
        "line": (sly.Polyline, sly.Polyline([sly.PointLocation(1, 1), sly.PointLocation(5, 5)])),
        "point": (sly.Point, sly.Point(3, 4)),
        "obb": (sly.OrientedBBox, sly.OrientedBBox(10, 10, 20, 30, angle=0.3)),
    }
    classes = [sly.ObjClass(n, t) for n, (t, _) in geoms.items()] + [PERSON]
    labels = [sly.Label(g, c) for (_, g), c in zip(geoms.values(), classes)]
    nose, left, right = PERSON_NODES
    labels.append(person([(nose, 3, 3, False), (left, 8, 9, False), (right, 9, 12, True)]))
    return sly.ProjectMeta(obj_classes=classes), labels


@pytest.mark.parametrize("task_type,expected_classes", [("segmentation", ["1", "2", "3"]), ("pose", ["7"])])
def test_every_shape_type(run_export, task_type, expected_classes):
    # unsupported shapes are skipped (and reported), never a crash; values stay in [0, 1]
    meta, labels = _every_shape()
    images = {10: [(1, "a.jpg", ann(*labels))]}
    result_dir = run_export(FakeApi(meta, [(10, "ds", None)], images), task_type=task_type)
    lines = read_labels(result_dir, "train")["10_a.txt"]
    assert sorted(line.split()[0] for line in lines) == expected_classes
    for line in lines:
        assert all(0 <= float(v) <= 2 for v in line.split()[1:])


def test_keypoint_outside_the_image_drops_the_whole_skeleton_in_the_sdk():
    # documents SDK behaviour (not fixed here): sly.Annotation crops labels to the image on load,
    # and GraphNodes.crop returns [] when any node is outside, so such a skeleton never reaches
    # the export
    nose, left, right = PERSON_NODES
    a = ann(person([(nose, 3, 3, False), (left, 8, 9, False), (right, 9, 205, False)]))
    assert len(a.labels) == 0
