# export-to-yolov8: fixes 2026-10

Branch `ecosystem-fixes-2026-10`, from `master` de45265 (v1.2.3). Lifecycle `maintained`:
targeted fixes only (same structure, `sly.app.Export`, same modal), plus an SDK bump that fixes a
crash (see Image).

## Behaviour contract

- **Launch**: context menu "Download as" on an images project or an images dataset.
- **Input**: project; optional dataset (now: that dataset plus every dataset nested in it).
- **Options**: `modal.state.taskType` = `segmentation` (default) | `pose`;
  `modal.state.includeVisibility` (default true in the modal, false when absent) adds the
  visibility flag in pose mode.
- **Output**: `<project id>_<project name>` tar via `sly.app.Export` (`set_download`), Team Files
  `/tmp/supervisely/export/Export to YOLOv8 format/<task id>/…`; workflow input = project, output =
  file. Inside:
  - `images/{train,val}/<dataset id>_<image name>`, `labels/{train,val}/<same stem>.txt`;
  - segmentation: `class x1 y1 x2 y2 …` per Polygon / Bitmap / AlphaMask (a bitmap with several
    parts becomes one line with the parts' contours concatenated); other shapes skipped;
  - pose: `class xc yc w h` + per keypoint `x y [v]` (v = 2 visible, 1 disabled, 0 missing),
    padded with zeros to the largest keypoint template in the project; other shapes skipped;
  - `data_config.yaml`: `train`, `val`, `nc`, `names`, `colors` (+ `kpt_shape` in pose mode).
  - Skipped labels are counted in one warning at the end; the task still succeeds.
- **Class index** = position in the project meta (all classes).
- **Split**: tag `val` → val; tag `train` or no tag → train; both tags → both splits.

## Fixed

| # | Commit | Problem | Tests old → new |
|---|---|---|---|
| 1 | Drop distutils: parse includeVisibility without strtobool | `from distutils.util import strtobool` (removed in Python 3.12); works in the 3.14 image only through setuptools' shim, fails in any env without it | 0/1 → 1/1 |
| 2 | Fix merged label lines when two images differ only in extension | `a.jpg`/`a.png` in one dataset shared one `.txt`, appended without a newline: one broken line for both | 0/2 → 2/2 |
| 3 | Cut over-long file names to the 255-byte file system limit | `<ds id>_<long non-ASCII name>` over 255 bytes: `File name too long`, export crashed | 0/1 → 1/1 |
| 4 | Export nested datasets when a dataset is selected | only the selected dataset's own images were exported; dev 1317 "Tropical" (0 own, 16 nested images) gave an archive with only `data_config.yaml`, as a success | 0/1 → 1/1 |
| 5 | Pose: write keypoints in class template order, keep slots of missing nodes | keypoints written in label storage order; a missing node shifted later keypoints into wrong slots | 0/3 → 3/3 |
| 6 | Move to SDK 6.74.34 on base-py-sdk-hardened | SDK 6.73.564 cannot parse a `multipolygon` class: crash on any project with one (dev 9390) | 0/1 → 1/1 (6.73.564 vs 6.74.34) |

Also: offline tests (`tests/`, fake `sly.Api`, 18 tests) including every image shape type.

## Intentional changes (visible to users)

- Label files end with `\n`. Old vs new on dev project 1317 (22 images): 45 files identical apart
  from that.
- On a stem clash the second image becomes `<stem>_1.<ext>`; names over 255 bytes are cut. Other
  names unchanged.
- Dataset export includes nested datasets (as convert-supervisely-to-yolov5-format v2.2.0; README
  updated).
- Pose lines follow the class template's node order; a node absent from a label is `0 0 [0]` in its
  own slot.
- `includeVisibility` values other than the strtobool truthy set mean false instead of raising.
- `instance_version` 6.15.60 → 6.17.22 (see Image).

## Found, not fixed

- SDK (both 6.73.564 and 6.74.34): `sly.Annotation` crops labels to the image on load and
  `GraphNodes.crop` returns nothing if any node is outside, so such skeletons silently vanish from
  a pose export (pinned in `tests/test_geometries.py`).
- A bitmap with several parts is exported as one polygon of concatenated contours (jumps between
  parts); polygon holes are dropped; rectangles and multipolygons are skipped in segmentation mode
  instead of converted.
- If all labels are skipped (e.g. a rectangle-only project in segmentation mode) the task still
  succeeds with empty label files and one warning.
- An annotation that fails to parse is replaced by an empty one (images exported as background).
- Pose mode mixes keypoint classes with different templates under one `kpt_shape` (padding);
  no `flip_idx`.
- Image tagged both `train` and `val` is in both splits.
- Bbox centre is floored (`Rectangle.center`), up to 1 px shift, as in the SDK.
- Security: no archive extraction, eval/pickle, shell or secrets; nothing in
  `ecosystem-fleet/audits/ecosystem-security.md` for this repo.

## SDK `to_yolo` divergences (noted, not fixed)

`supervisely/convert/image/yolo/yolo_helper.py` (origin/master):
- `keypoints_to_yolo_line` has the same label-order / missing-node bug fixed here (#5), always
  writes 3 values per keypoint and pads per class template; `save_yolo_config` writes `kpt_shape
  [n, 3]` and an identity `flip_idx`.
- Segment: SDK writes one line per polygon of a multi-part bitmap (separate instances); the app
  concatenates parts into one line. SDK converts AnyGeometry labels, skips the rest like the app.
- SDK supports detection; this app does not (points to the YOLOv5 app).
- File names: SDK `<dataset short name>_<image>` with `generate_free_name` on full names (label
  stems of `a.jpg`/`a.png` still collide and overwrite); the app uses the dataset id.
- Number format `:.6f` (SDK) vs `round(v, 6)` (app); train/val: SDK puts a both-tagged image in
  val only.

## Image

`supervisely/base-py-sdk-hardened:6.73.564` → `supervisely/base-py-sdk-hardened:6.74.34` (Python
3.14.7, tag exists), `supervisely==6.74.34`, `instance_version` 6.15.60 → 6.17.22 (`versions.json`
allows [6.74.29, 6.74.35) for 6.17.22). Same pair as export-to-coco and download-images. Needed to
read projects with multipolygon classes. It drops instances older than 6.17.22: for a maintained
app this is a product call; revert that single commit to keep 6.15.60 (fixes 1–5 do not depend on
it). No extra dependencies; `release.yml` has no build-image job.

## Verification

- Unit: 18/18 in a Python 3.14.3 venv with supervisely 6.74.34 and inside
  `base-py-sdk-hardened:6.74.34` (3.14.7, amd64); fixes 1–5 also pass on 6.73.564.
- Read-only run against dev (`MyExport.process` with a real `sly.Api`, no upload):
  - 1317 project, segmentation: old (6.73.564) and new give the same 45 files (newline only);
  - 1317 dataset 10122: old 0 images, new 16;
  - 9390 pose: old crashes (`Unknown shape: 'multipolygon'`); new 2 images, one `kp` line each,
    15 labels skipped and reported;
  - 9390 segmentation: new 2 images, 5 + 2 lines (polygons, bitmap), 10 skipped.

## Dev-check plan

Run from `workapps/`; each case on the branch, then on the published version.

```
# 1. Nested project, segmentation (rectangle-only project)
uv run ecosystem-fleet/fleet.py run export-to-yolov8 --workspace 3 --project 1317 --state taskType=segmentation --version ecosystem-fixes-2026-10 --branch
uv run ecosystem-fleet/fleet.py run export-to-yolov8 --workspace 3 --project 1317 --state taskType=segmentation
```
Expected: success, 22 images in train, 22 empty label files, warning "… labels skipped …
{'Apple': 'rectangle', …}". Published: same file list (labels without the final newline).

```
# 2. Parent dataset whose images are all in nested datasets
uv run ecosystem-fleet/fleet.py run export-to-yolov8 --workspace 3 --project 1317 --dataset 10122 --version ecosystem-fixes-2026-10 --branch
uv run ecosystem-fleet/fleet.py run export-to-yolov8 --workspace 3 --project 1317 --dataset 10122
```
Expected: 16 images (`10123_*` … `10127_*`). Published: success with no images (only
`data_config.yaml`).

```
# 3. Every shape type, pose
uv run ecosystem-fleet/fleet.py run export-to-yolov8 --workspace 733 --project 9390 --state taskType=pose --state includeVisibility=true --version ecosystem-fixes-2026-10 --branch
uv run ecosystem-fleet/fleet.py run export-to-yolov8 --workspace 733 --project 9390 --state taskType=pose --state includeVisibility=true
```
Expected: success; `48091_a.png`, `48092_b.png`, one line each: class 2 (`kp`), bbox, 2 keypoints
with `v=2`; `kpt_shape: [2, 3]`; warning lists 15 skipped labels. Published: fails with
`Unknown shape: 'multipolygon'`.

```
# 4. Every shape type, segmentation
uv run ecosystem-fleet/fleet.py run export-to-yolov8 --workspace 733 --project 9390 --state taskType=segmentation --version ecosystem-fixes-2026-10 --branch
uv run ecosystem-fleet/fleet.py run export-to-yolov8 --workspace 733 --project 9390 --state taskType=segmentation
```
Expected: success; 5 + 2 lines (classes 0 blob, 1 building, 6 roof), all values in [0, 1];
warning lists 10 skipped labels (line, point, graph, multipolygon). Published: fails as in case 3.
