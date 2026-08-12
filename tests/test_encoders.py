"""Unit tests for the pluggable image / language encoders.

The ViT and CLIP tests download two *tiny* random-weight checkpoints from the HF hub
(a few hundred KB each) and are skipped when the hub is unreachable.
"""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from robometer_policy_learning.modules.encoders import (
    FEATURIZER_IMAGE_ENCODER_TYPES,
    CLIPLangEncoder,
    ViTImageFeaturizer,
    build_image_featurizer,
    build_language_encoder,
    is_featurizer_image_encoder,
    language_embedding_dim,
)

TINY_VIT = "hf-internal-testing/tiny-random-ViTModel"
TINY_CLIP = "hf-internal-testing/tiny-random-CLIPModel"


def _hub_available() -> bool:
    try:  # a tiny metadata call is enough to know whether we can pull checkpoints
        from huggingface_hub import model_info

        model_info(TINY_VIT)
        return True
    except Exception:
        return False


needs_hub = pytest.mark.skipif(not _hub_available(), reason="HF hub unreachable (offline)")


@pytest.fixture(scope="module")
def images():
    """A batch of uint8 (B, H, W, C) images, the layout the envs produce."""
    return torch.randint(0, 255, (2, 84, 84, 3), dtype=torch.uint8)


# ---------------------------------------------------------------------------
# Type registry / factory (no downloads)
# ---------------------------------------------------------------------------

def test_vit_is_a_registered_featurizer_type():
    assert "vit" in FEATURIZER_IMAGE_ENCODER_TYPES
    assert is_featurizer_image_encoder("vit")
    assert is_featurizer_image_encoder("VIT")  # case-insensitive
    assert not is_featurizer_image_encoder(None)
    assert not is_featurizer_image_encoder("flatten")


def test_unknown_image_encoder_type_lists_supported_types():
    with pytest.raises(ValueError, match="vit"):
        build_image_featurizer("swin", input_shape=(84, 84, 3))


def test_unknown_language_encoder_type():
    with pytest.raises(ValueError, match="clip"):
        build_language_encoder("word2vec")


def test_language_embedding_dim_falls_back():
    assert language_embedding_dim(object(), default=384) == 384


# ---------------------------------------------------------------------------
# ViT image featurizer
# ---------------------------------------------------------------------------

@needs_hub
@pytest.mark.parametrize("pool", ["cls", "mean", "pooler"])
def test_vit_pooling_modes_return_batched_features(images, pool):
    enc = build_image_featurizer("vit", input_shape=(84, 84, 3), vit_model=TINY_VIT, vit_pool=pool)
    out = enc(images)
    assert out.shape == (images.shape[0], enc.output_dim)
    assert out.dtype == torch.float32


@needs_hub
def test_vit_rejects_unknown_pool():
    with pytest.raises(ValueError, match="cls|mean|pooler"):
        ViTImageFeaturizer(vit_model=TINY_VIT, pool="max")


@needs_hub
def test_vit_accepts_channels_first_float_and_frame_dims(images):
    enc = ViTImageFeaturizer(vit_model=TINY_VIT)
    chw = images.permute(0, 3, 1, 2).float() / 255.0
    assert enc(chw).shape == (2, enc.output_dim)
    assert enc(images.unsqueeze(1)).shape == (2, enc.output_dim)  # (B, 1, H, W, C)
    grayscale = torch.rand(2, 1, 84, 84)
    assert enc(grayscale).shape == (2, enc.output_dim)


@needs_hub
def test_vit_projection_sets_output_dim(images):
    enc = ViTImageFeaturizer(vit_model=TINY_VIT, output_dim=16)
    assert enc.output_dim == 16
    assert enc(images).shape == (2, 16)


@needs_hub
def test_frozen_vit_has_no_grads_and_stays_in_eval(images):
    enc = ViTImageFeaturizer(vit_model=TINY_VIT, finetune=False)
    enc.train()  # even when the parent module is switched to train mode
    assert not enc.vit.training
    assert not any(p.requires_grad for p in enc.vit.parameters())
    assert not enc(images).requires_grad


@needs_hub
def test_finetuned_vit_propagates_gradients(images):
    enc = ViTImageFeaturizer(vit_model=TINY_VIT, output_dim=8, finetune=True)
    enc(images).sum().backward()
    assert any(p.grad is not None for p in enc.vit.parameters())


@needs_hub
def test_vit_interpolates_position_embeddings_for_other_input_sizes(images):
    native = ViTImageFeaturizer(vit_model=TINY_VIT)
    resized = ViTImageFeaturizer(vit_model=TINY_VIT, image_size=native.target_size * 2)
    assert not native._interpolate_pos_encoding
    assert resized._interpolate_pos_encoding
    assert resized(images).shape == (2, resized.output_dim)


@needs_hub
def test_clip_checkpoint_reduces_to_its_vision_tower(images):
    enc = ViTImageFeaturizer(vit_model=TINY_CLIP)
    assert not hasattr(enc.vit, "text_model")  # only the vision tower is kept
    assert enc(images).shape == (2, enc.output_dim)


@needs_hub
def test_vit_backbone_is_shared_across_camera_keys():
    from robometer_policy_learning.utils.featurizers import ObservationFeaturizer

    obs_space = spaces.Dict(
        {
            "agentview_image": spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8),
            "eye_in_hand_image": spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8),
            "state": spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32),
        }
    )
    featurizer = ObservationFeaturizer(
        obs_space,
        featurizer_cfg={"state": [32]},
        image_encoder_type="vit",
        vit_model=TINY_VIT,
        vit_projection_dim=16,
    )
    left = featurizer.image_encoders["agentview_image"]
    right = featurizer.image_encoders["eye_in_hand_image"]
    assert left.vit is right.vit  # one backbone, loaded once
    assert left.projection is not right.projection  # but a per-camera projection

    obs = {
        "agentview_image": torch.randint(0, 255, (2, 84, 84, 3), dtype=torch.uint8),
        "eye_in_hand_image": torch.randint(0, 255, (2, 84, 84, 3), dtype=torch.uint8),
        "state": torch.randn(2, 10),
    }
    assert featurizer.flatten_obs(obs).shape == (2, featurizer.output_dim)
    assert featurizer.output_dim == 16 + 16 + 32


# ---------------------------------------------------------------------------
# CLIP language encoder
# ---------------------------------------------------------------------------

@needs_hub
def test_clip_lang_encoder_embeds_batches_of_instructions():
    enc = build_language_encoder("clip", model_name=TINY_CLIP)
    assert isinstance(enc, CLIPLangEncoder)
    emb = enc.get_lang_emb(["pick up the black bowl", "close the drawer"])
    assert emb.shape == (2, enc.embedding_dim)
    assert emb.dtype == torch.float32
    assert enc.get_lang_emb("a single instruction").shape == (1, enc.embedding_dim)


@needs_hub
def test_clip_lang_encoder_is_sentence_transformer_compatible():
    """``encode`` must behave like ``SentenceTransformer.encode`` (env precompute path)."""
    enc = build_language_encoder("clip", model_name=TINY_CLIP)
    single = enc.encode("pick up the bowl")
    assert isinstance(single, np.ndarray) and single.shape == (enc.embedding_dim,)
    batch = enc.encode(["a", "b", "c"], convert_to_tensor=True)
    assert isinstance(batch, torch.Tensor) and batch.shape == (3, enc.embedding_dim)
    # batching must not change the result
    torch.testing.assert_close(batch, enc.encode(["a", "b", "c"], batch_size=1, convert_to_tensor=True))


@needs_hub
def test_clip_lang_encoder_normalization_is_per_call():
    enc = build_language_encoder("clip", model_name=TINY_CLIP)
    normalized = enc.encode(["a"], convert_to_tensor=True, normalize_embeddings=True)
    assert float(normalized.norm()) == pytest.approx(1.0, abs=1e-5)
    assert enc.normalize is False  # the encoder's own setting is restored


@needs_hub
def test_clip_projection_toggle_changes_embedding_dim():
    projected = build_language_encoder("clip", model_name=TINY_CLIP, use_projection=True)
    hidden = build_language_encoder("clip", model_name=TINY_CLIP, use_projection=False)
    assert projected.embedding_dim != hidden.embedding_dim
    assert hidden.embedding_dim == hidden.model.config.hidden_size


@needs_hub
def test_transformer_feature_extractor_uses_clip_language_encoder():
    from robometer_policy_learning.modules.transformer.transformer_utils import TransformerFeatureExtractor

    obs_space = spaces.Dict(
        {
            "image": spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8),
            "state": spaces.Box(-np.inf, np.inf, (10,), dtype=np.float32),
        }
    )
    extractor = TransformerFeatureExtractor(
        observation_space=obs_space,
        featurizer_cfg={"state": [32]},
        image_encoder_type="vit",
        vit_model=TINY_VIT,
        vit_projection_dim=24,
        use_language_embeddings=True,
        lang_encoder_type="clip",
        lang_model_name=TINY_CLIP,
    )
    assert isinstance(extractor.lang_encoder, CLIPLangEncoder)
    # the extractor adopts the encoder's real width instead of the MiniLM default
    assert extractor.lang_embedding_dim == extractor.lang_encoder.embedding_dim
    assert extractor.obs_dim == 24 + 32

    lang = extractor._encode_language_if_needed(["pick up the bowl", "open the drawer"])
    assert lang.shape == (2, extractor.lang_embedding_dim)
    assert "pick up the bowl" in extractor.lang_embedding_cache  # cached for reuse
