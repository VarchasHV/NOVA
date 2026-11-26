"""The network code lives in `submission/models/`, so that `submission/` works on its own.

This package only points there, which keeps `import models` working for training, evaluation and
tools. Put no code here.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1] / "submission" / "models")]
