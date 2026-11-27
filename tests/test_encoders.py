import numpy as np
import pytest
import torch

from doom_env import make_env
from helpers import random_obs
from models import obs as O
from models import policy as P
from models.actor import BBBActor
from models.config import ModelConfig
from models.encoders import EntityEncoder, ObservationEncoder


@pytest.mark.parametrize("use_depth", [False, True])
def test_encoder_shapes(use_depth):
    cfg = ModelConfig(use_depth=use_depth)
    assert ObservationEncoder(cfg)(random_obs(4, use_depth)).shape == (4, cfg.d_model)


def test_no_entities_is_finite_and_has_gradients():
    enc = EntityEncoder()
    obs = random_obs(3, entities=0)
    out = enc(obs["ent_feat"], obs["ent_type"], obs["ent_mask"])
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in enc.parameters() if p.grad is not None)


def test_padding_entities_do_not_change_the_output():
    enc = EntityEncoder().eval()
    obs = random_obs(2, entities=3)
    a = enc(obs["ent_feat"], obs["ent_type"], obs["ent_mask"])
    obs["ent_feat"][:, 3:] = 99.0
    b = enc(obs["ent_feat"], obs["ent_type"], obs["ent_mask"])
    torch.testing.assert_close(a, b)


def test_preprocess_cleans_bad_input():
    env = make_env("maps/two_rooms.wad")
    obs, _ = env.reset(seed=0)
    env.close()
    obs["vitals"][0] = np.nan
    obs["rays"]["distance"][3] = np.inf
    obs["rays"]["hit_type"][4] = 99
    compact = O.preprocess(obs, np.zeros(9), use_depth=True)
    for name, (shape, dtype) in O.obs_spec(True).items():
        assert compact[name].shape == shape and compact[name].dtype == dtype, name
        assert np.isfinite(compact[name]).all(), name
    assert compact["ray_type"].max() < O.NUM_HIT_TYPES


def test_action_heads():
    logits = torch.randn(7, P.NUM_LOGITS)
    idx = P.sample(logits)
    buttons = P.to_buttons(idx)
    assert buttons.shape == (7, 9) and set(buttons.unique().tolist()) <= {0.0, 1.0}
    for a, b in ((0, 1), (2, 3), (4, 5)):  # opposite buttons are never pressed together
        assert not ((buttons[:, a] == 1) & (buttons[:, b] == 1)).any()
    assert P.log_prob(logits, idx).shape == (7,)
    assert (P.entropy(logits) >= 0).all()


def test_actor_rejects_privileged_input():
    actor = BBBActor(ModelConfig())
    obs = random_obs(1)
    obs["priv"] = torch.zeros(1, 3)
    with pytest.raises(ValueError, match="must not receive"):
        actor.step(obs, actor.initial_state(1))
