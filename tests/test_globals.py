import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Python 3.12 removed distutils; the hardened image only has it through setuptools' shim
PROBE = """
import sys
sys.modules["distutils"] = None  # what a 3.12+ interpreter without setuptools' shim sees
sys.modules["distutils.util"] = None
import src.globals as g
print(g.IS_POSE_EST_TASK, g.INCLUDE_VISIBILTY_FLAG)
"""


def _run(task_type, include_visibility):
    env = dict(os.environ, ENV="development")
    env["modal.state.taskType"] = task_type
    if include_visibility is None:
        env.pop("modal.state.includeVisibility", None)
    else:
        env["modal.state.includeVisibility"] = include_visibility
    proc = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout.strip().splitlines()[-1]


def test_globals_import_without_distutils():
    assert _run("pose", "true") == "True True"
    assert _run("pose", "false") == "True False"
    assert _run("pose", "True") == "True True"
    assert _run("pose", None) == "True False"
    assert _run("segmentation", "true") == "False False"
