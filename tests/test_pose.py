from conftest import FakeApi, read_labels
from helpers import META, PERSON_NODES, ann, person

NOSE, LEFT, RIGHT = PERSON_NODES


def _keypoints(line, dims=3):
    values = line.split()[5:]
    return [values[i : i + dims] for i in range(0, len(values), dims)]


def test_keypoints_follow_the_class_template_order(run_export):
    # YOLO pose reads keypoint k of every line as the same body part. The label stores its
    # nodes in another order than the class template (right, nose, left)
    p = person([(RIGHT, 50, 60, False), (NOSE, 10, 20, False), (LEFT, 30, 40, False)])
    images = {10: [(1, "a.jpg", ann(p))]}
    result_dir = run_export(FakeApi(META, [(10, "ds", None)], images), task_type="pose")
    (line,) = read_labels(result_dir, "train")["10_a.txt"]
    assert _keypoints(line) == [["0.1", "0.1", "2"], ["0.2", "0.3", "2"], ["0.3", "0.5", "2"]]


def test_missing_node_keeps_its_slot(run_export):
    # the label has no "left" node: it must be written as not labelled in its own slot, not
    # shift "right" into the "left" slot
    p = person([(NOSE, 10, 20, False), (RIGHT, 50, 60, False)])
    images = {10: [(1, "a.jpg", ann(p))]}
    result_dir = run_export(FakeApi(META, [(10, "ds", None)], images), task_type="pose")
    (line,) = read_labels(result_dir, "train")["10_a.txt"]
    assert _keypoints(line) == [["0.1", "0.1", "2"], ["0", "0", "0"], ["0.3", "0.5", "2"]]


def test_missing_node_without_visibility(run_export):
    p = person([(NOSE, 10, 20, False), (RIGHT, 50, 60, False)])
    images = {10: [(1, "a.jpg", ann(p))]}
    result_dir = run_export(
        FakeApi(META, [(10, "ds", None)], images), task_type="pose", include_visibility=False
    )
    (line,) = read_labels(result_dir, "train")["10_a.txt"]
    assert _keypoints(line, dims=2) == [["0.1", "0.1"], ["0", "0"], ["0.3", "0.5"]]
