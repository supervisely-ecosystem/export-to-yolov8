from conftest import FakeApi, list_images, read_labels
from helpers import META, ann, poly


def test_same_stem_different_extension_keeps_labels_apart(run_export):
    # a.jpg and a.png in one dataset used to share labels/train/10_a.txt; the file was opened in
    # append mode without a trailing newline, so both images got one merged, broken line
    images = {
        10: [
            (1, "a.jpg", ann(poly([(10, 20), (10, 60), (50, 60)]))),
            (2, "a.png", ann(poly([(20, 20), (20, 60), (60, 60)]))),
        ]
    }
    result_dir = run_export(FakeApi(META, [(10, "ds", None)], images))
    imgs = list_images(result_dir, "train")
    labels = read_labels(result_dir, "train")
    assert len(imgs) == 2
    assert len(labels) == 2
    assert sorted(n.rsplit(".", 1)[0] for n in imgs) == sorted(n[: -len(".txt")] for n in labels)
    for lines in labels.values():
        assert len(lines) == 1 and len(lines[0].split()) == 7


def test_label_files_end_with_newline(run_export):
    images = {10: [(1, "a.jpg", ann(poly([(10, 20), (10, 60), (50, 60)]), poly([(1, 1), (1, 5), (5, 5)])))]}
    result_dir = run_export(FakeApi(META, [(10, "ds", None)], images))
    with open(f"{result_dir}/labels/train/10_a.txt") as f:
        text = f.read()
    assert text.endswith("\n") and text.count("\n") == 2

