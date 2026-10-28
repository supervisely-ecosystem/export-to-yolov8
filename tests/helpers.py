"""Tiny in-memory SDK objects for the tests."""

import numpy as np
import supervisely as sly
from supervisely.geometry.graph import GraphNodes, KeypointsTemplate, Node

H, W = 100, 200

POLY = sly.ObjClass("poly", sly.Polygon)
MASK = sly.ObjClass("mask", sly.Bitmap)
BOX = sly.ObjClass("box", sly.Rectangle)

_tmpl = KeypointsTemplate()
_tmpl.add_point("nose", row=0, col=0)
_tmpl.add_point("left", row=5, col=0)
_tmpl.add_point("right", row=5, col=10)
PERSON = sly.ObjClass("person", GraphNodes, geometry_config=_tmpl)
# node ids as the template stores them, in template order
PERSON_NODES = list(PERSON.geometry_config["nodes"].keys())

TRAIN = sly.TagMeta("train", sly.TagValueType.NONE)
VAL = sly.TagMeta("val", sly.TagValueType.NONE)
META = sly.ProjectMeta(obj_classes=[POLY, MASK, BOX, PERSON], tag_metas=[TRAIN, VAL])


def ann(*labels, tags=(), size=(H, W)):
    return sly.Annotation(
        img_size=size,
        labels=list(labels),
        img_tags=sly.TagCollection([sly.Tag({"train": TRAIN, "val": VAL}[t]) for t in tags]),
    )


def poly(points, cls=POLY):
    return sly.Label(sly.Polygon([sly.PointLocation(r, c) for r, c in points]), cls)


def mask(cls=MASK):
    return sly.Label(sly.Bitmap(np.ones((4, 4), bool), origin=sly.PointLocation(10, 20)), cls)


def rect(top, left, bottom, right, cls=BOX):
    return sly.Label(sly.Rectangle(top, left, bottom, right), cls)


def person(nodes, cls=PERSON):
    """nodes: [(node_id, row, col, disabled)] in the order the label stores them."""
    return sly.Label(
        GraphNodes({k: Node(sly.PointLocation(r, c), disabled=d) for k, r, c, d in nodes}), cls
    )
