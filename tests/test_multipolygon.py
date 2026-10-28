"""Projects with a Multipolygon class (instance 6.17+, e.g. dev project 9390) — raw JSON, because
SDK 6.73.x cannot even build such a class: ProjectMeta.from_json raised "Unknown shape:
'multipolygon'" and the export crashed before writing anything."""

from conftest import FakeApi, read_labels

META_JSON = {
    "classes": [
        {"title": "poly", "shape": "polygon", "color": "#FF0000", "geometry_config": {}},
        {"title": "parcel", "shape": "multipolygon", "color": "#0F7C8A", "geometry_config": {}},
    ],
    "tags": [],
}
ANN_JSON = {
    "size": {"height": 100, "width": 200},
    "tags": [],
    "description": "",
    "objects": [
        {"classTitle": "poly", "tags": [], "geometryType": "polygon",
         "points": {"exterior": [[20, 10], [60, 10], [60, 50]], "interior": []}},
        {"classTitle": "parcel", "tags": [], "geometryType": "multipolygon",
         "parts": [{"exterior": [[1, 1], [9, 1], [9, 9]], "interior": []},
                   {"exterior": [[20, 20], [40, 20], [40, 40]], "interior": []}]},
    ],
}


def test_project_with_a_multipolygon_class(run_export):
    # the export runs; the multipolygon is skipped and reported like other unsupported shapes
    api = FakeApi(META_JSON, [(10, "ds", None)], {10: [(1, "a.jpg", ANN_JSON)]})
    result_dir = run_export(api)
    assert read_labels(result_dir, "train")["10_a.txt"] == ["0 0.1 0.1 0.3 0.1 0.3 0.5"]
