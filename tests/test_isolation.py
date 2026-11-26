"""The submission must contain actor code only: no training or evaluation modules at inference."""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("training", "evaluation", "tools")


def test_importing_the_agent_loads_no_training_code():
    code = (
        "import sys\n"
        "import submission.agent as a\n"
        "from doom_env import spaces as S\n"
        "agent = a.DoomAgent(S.observation_space(), S.action_space(), 'cpu')\n"
        f"bad = [m for m in sys.modules if m.split('.')[0] in {FORBIDDEN!r}]\n"
        "assert not bad, bad\n"
        "assert 'vizdoom' not in sys.modules, 'the agent must not touch the engine'\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_actor_sources_never_import_training_code():
    for folder in ("models", "submission"):
        for path in (ROOT / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                        [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
                assert not [n for n in names if n.split(".")[0] in (*FORBIDDEN, "vizdoom")], path


def test_no_hand_written_aiming_or_pathfinding_in_actor_code():
    banned = ("atan2", "arctan", "dijkstra", "heapq")
    for folder in ("models", "submission"):
        for path in (ROOT / folder).rglob("*.py"):
            text = path.read_text().lower()
            assert not [w for w in banned if w in text and "aimbot" not in text], path


def test_submission_folder_works_on_its_own(tmp_path):
    """The scorer may take only `submission/` and run it inside the official template."""
    shutil.copytree(ROOT / "submission", tmp_path / "submission", ignore=shutil.ignore_patterns("__pycache__"))
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "import submission.agent as a\n"
        "from doom_env import spaces as S\n"
        "agent = a.DoomAgent(S.observation_space(), S.action_space(), 'cpu')\n"
        "import models.actor\n"
        f"assert models.actor.__file__.startswith({str(tmp_path.resolve())!r}), models.actor.__file__\n"
        "assert agent.act(agent._blank_observation()).shape == (9,) and agent.errors == 0\n"
    )
    proc = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
