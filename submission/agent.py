"""Streamed Mind: evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6. Every action comes from the trained network in `models/`. The code
around it is a safety shell: it cleans inputs, validates outputs and, only if something raises,
returns a simple moving action so the player is never frozen for the rest of a map.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = SUBMISSION_DIR / "weights" / "model.pt"
if str(SUBMISSION_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR.parent))

import torch  # noqa: E402

from models import obs as O  # noqa: E402
from models import policy as P  # noqa: E402
from models.actor import StreamedMindActor  # noqa: E402
from models.config import ModelConfig  # noqa: E402

SEED = 0


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        """Called once: load the weights and warm the network up."""
        torch.set_num_threads(1)
        self.observation_space = observation_space
        self.action_space = action_space
        # We run on the CPU even when a GPU is offered: for a ~1M parameter model deciding one step at a
        # time it is faster (RTX 3050 laptop, while training: 2.3 ms CPU vs 10 ms GPU mean per act).
        self.device = torch.device("cpu")
        self.mode, self.temperature = "sample", 1.0
        self.errors = 0
        self.weights_loaded = False
        self.model = self._load()
        self.reset()
        try:  # first forward pass allocates kernels and, on a GPU, initialises CUDA
            self._decide(self._blank_observation())
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        self.reset()

    def _load(self) -> StreamedMindActor:
        try:
            if WEIGHTS_PATH.is_file():
                data = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
                model = StreamedMindActor(ModelConfig.from_dict(data["model_config"]))
                model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in data["actor"].items()})
                self.mode = data.get("inference", {}).get("mode", self.mode)
                self.temperature = float(data.get("inference", {}).get("temperature", self.temperature))
                self.weights_loaded = True
                return model.to(self.device).eval()
            print(f"WARNING: no weights at {WEIGHTS_PATH}; using an UNTRAINED network", flush=True)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            print(f"WARNING: could not load {WEIGHTS_PATH} (Git LFS pointer?); using an UNTRAINED network", flush=True)
        torch.manual_seed(SEED)
        return StreamedMindActor(ModelConfig()).to(self.device).eval()

    def _blank_observation(self) -> dict:
        return {
            "rays": {"distance": np.ones(O.NUM_RAYS, np.float32), "hit_type": np.zeros(O.NUM_RAYS, np.int64)},
            "visible_entities": {
                "features": np.zeros((O.MAX_ENTITIES, O.ENTITY_FEATURES), np.float32),
                "entity_type": np.zeros(O.MAX_ENTITIES, np.int64),
                "mask": np.zeros(O.MAX_ENTITIES, np.int8),
            },
            "vitals": np.zeros(O.VITALS_DIM, np.float32),
            "depth": np.zeros((120, 160), np.uint8),
        }

    def reset(self) -> None:
        """Called before every map: forget everything."""
        self.state = None
        self.prev_action = np.zeros(O.NUM_BUTTONS, np.float32)
        self.calls = 0
        self.rng = torch.Generator().manual_seed(SEED)
        self.last_info: dict = {}
        self.last_pose = None

    def _decide(self, observation: dict) -> np.ndarray:
        compact = O.preprocess(observation, self.prev_action, self.model.cfg.use_depth)
        tensors = {k: torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(0).to(self.device) for k, v in compact.items()}
        with torch.inference_mode():
            if self.state is None:
                self.state = self.model.initial_state(1, self.device)
            out = self.model.step(tensors, self.state)
            if not all(bool(torch.isfinite(v).all()) for v in out.state.values()) or not bool(torch.isfinite(out.logits).all()):
                self.state = None  # corrupted memory: start it fresh next decision
                raise FloatingPointError("non-finite network state")
            self.state = out.state
            self.last_info, self.last_pose = out.info, out.pose
            logits = out.logits.cpu()
            idx = P.greedy(logits) if self.mode == "greedy" else P.sample(logits, self.temperature, self.rng)
            return P.to_buttons(idx)[0].numpy()

    def _fallback(self) -> np.ndarray:
        action = np.zeros(O.NUM_BUTTONS, np.int8)
        action[0] = 1  # forward
        action[4 if (self.calls // 20) % 2 == 0 else 5] = 1  # turn, alternating so it does not circle
        return action

    def act(self, observation: dict) -> np.ndarray:
        """Return a MultiBinary action: int8 array of shape (9,) with values in {0, 1}."""
        self.calls += 1
        try:
            action = np.asarray(self._decide(observation))
            if action.shape != (O.NUM_BUTTONS,) or not np.isfinite(action).all():
                raise ValueError(f"bad action {action!r}")
            action = (action > 0.5).astype(np.int8)
        except Exception:  # noqa: BLE001 - a crash would freeze the player for the whole map
            self.errors += 1
            if self.errors <= 3:
                traceback.print_exc()
            action = self._fallback()
        self.prev_action = action.astype(np.float32)
        return action
