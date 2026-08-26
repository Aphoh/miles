from argparse import Namespace
from types import SimpleNamespace

from miles.backends.megatron_utils.model_provider import _apply_bridge_yarn_config


def test_explicit_yarn_config_overrides_bridge_provider():
    provider = SimpleNamespace(seq_length=40960, position_embedding_type="rope")
    args = Namespace(
        position_embedding_type="yarn",
        seq_length=131072,
        rotary_scaling_factor=4.0,
        yarn_original_max_position_embeddings=32768,
        yarn_beta_fast=None,
        yarn_beta_slow=None,
        mscale=None,
        mscale_all_dim=None,
        yarn_correction_range_round_to_int=None,
    )

    _apply_bridge_yarn_config(provider, args)

    assert provider.seq_length == 131072
    assert provider.position_embedding_type == "yarn"
    assert provider.yarn_rotary_scaling_factor == 4.0
    assert provider.yarn_original_max_position_embeddings == 32768
    assert provider.yarn_beta_fast == 32.0
    assert provider.yarn_beta_slow == 1.0
    assert provider.yarn_mscale == 1.0
    assert provider.yarn_mscale_all_dim == 0.0
    assert provider.yarn_correction_range_round_to_int is True


def test_non_yarn_config_preserves_bridge_provider():
    provider = SimpleNamespace(seq_length=40960, position_embedding_type="rope")
    args = Namespace(position_embedding_type="rope")

    _apply_bridge_yarn_config(provider, args)

    assert vars(provider) == {
        "seq_length": 40960,
        "position_embedding_type": "rope",
    }


def test_mla_yarn_config_preserves_bridge_provider():
    provider = SimpleNamespace(seq_length=4096, position_embedding_type="rope")
    args = Namespace(position_embedding_type="yarn", multi_latent_attention=True)

    _apply_bridge_yarn_config(provider, args)

    assert vars(provider) == {
        "seq_length": 4096,
        "position_embedding_type": "rope",
    }
