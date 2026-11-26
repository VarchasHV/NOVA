"""Architecture settings. Saved inside every checkpoint so the agent rebuilds the exact network."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    d_model: int = 256  # fused observation feature
    core_size: int = 256  # recurrent core
    two_timescale: bool = False  # add a slow GRU that updates every `slow_every` decisions
    slow_every: int = 8
    use_depth: bool = False  # depth image encoder (rays already give 360 degree geometry)
    # --- streamed spatial memory -------------------------------------------------------------
    memory: bool = True
    memory_mode: str = "anchored"  # "anchored" (position-anchored writes) or "soft" (fully learned)
    slots: int = 32
    slot_dim: int = 64
    heads: int = 4
    readout_dim: int = 128
    new_slot_radius: float = 320.0  # map units; farther than this from every slot -> new slot
    # --- auxiliary head (training signal only, unused for actions) ---------------------------
    aux_dim: int = 4

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown model settings: {sorted(unknown)}")
        return cls(**d)
