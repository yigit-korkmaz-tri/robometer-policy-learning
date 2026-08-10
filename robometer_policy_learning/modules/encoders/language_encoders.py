"""
Pluggable language-instruction encoders.

These turn task strings ("pick up the black bowl ...") into a fixed-size embedding that is
concatenated with the other observation features. Two backends are provided:

  * :class:`MiniLMLangEncoder` - sentence-transformers MiniLM (the historical default, 384-d).
  * :class:`CLIPLangEncoder`   - the text tower of a CLIP checkpoint (512-d for ViT-B/32),
    which pairs naturally with a CLIP/ViT image encoder since both land in the same space.

Contract for every encoder here (deliberately ``SentenceTransformer``-compatible, so these
can be dropped into the env-level precompute path as well as into the actor):
  * ``get_lang_emb(strings) -> (B, D)`` tensor on the encoder's device.
  * ``encode(strings, convert_to_tensor=..., device=...)`` mirrors ``SentenceTransformer.encode``
    (a single string yields a 1-D result, a list yields ``(B, D)``).
  * ``.embedding_dim`` reports the embedding size.

The encoders are frozen feature extractors (``eval()`` + ``no_grad``); they are intentionally
plain objects rather than ``nn.Module`` so their weights are neither optimized nor written into
actor checkpoints.
"""

from typing import Any, List, Optional, Sequence, Union

import torch

LANGUAGE_ENCODER_TYPES = ("minilm", "sentence_transformer", "clip")

MINILM_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_DEFAULT_MODEL = "openai/clip-vit-base-patch32"


class MiniLMLangEncoder:
    """Language encoder using MiniLM model for generating embeddings."""

    def __init__(self, device="cpu", model_name=MINILM_DEFAULT_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for language encoding. "
                "Please install it with: pip install sentence-transformers"
            )

        self.device = device
        self.model = SentenceTransformer(model_name)
        self.model.to(device)

        # Provide .encode for compatibility with some callers
        self.encode = self.model.encode

    @property
    def embedding_dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def get_lang_emb(self, lang_strings):
        """Get language embeddings for a list of language strings."""
        if isinstance(lang_strings, str):
            lang_strings = [lang_strings]

        embeddings = self.model.encode(lang_strings, convert_to_tensor=True, device=self.device)
        return embeddings


class CLIPLangEncoder:
    """Language encoder using the text tower of a CLIP checkpoint.

    Args:
        device: device the text tower runs on.
        model_name: HF CLIP model id (e.g. ``openai/clip-vit-base-patch32``).
        use_projection: when True (default) return the projected text embedding that lives in
            CLIP's shared image/text space (``projection_dim``, 512 for ViT-B/32); when False
            return the text transformer's pooled hidden state (``hidden_size``).
        normalize: L2-normalize the embeddings, as CLIP does before computing similarities.
        max_length: tokenizer truncation length; defaults to the checkpoint's max positions.
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cpu",
        model_name: str = CLIP_DEFAULT_MODEL,
        use_projection: bool = True,
        normalize: bool = False,
        max_length: Optional[int] = None,
    ):
        try:
            from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
        except ImportError as e:
            raise ImportError("transformers is required for the CLIP language encoder") from e

        self.device = device
        self.model_name = model_name
        self.use_projection = use_projection
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_cls = CLIPTextModelWithProjection if use_projection else CLIPTextModel
        self.model = model_cls.from_pretrained(model_name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.max_length = int(max_length or getattr(self.model.config, "max_position_embeddings", 77))
        hidden_size = int(self.model.config.hidden_size)
        if use_projection:
            self._embedding_dim = int(getattr(self.model.config, "projection_dim", None) or hidden_size)
        else:
            self._embedding_dim = hidden_size

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def to(self, device: Union[str, torch.device]) -> "CLIPLangEncoder":
        self.device = device
        self.model.to(device)
        return self

    def _embed(self, texts: List[str], device: Union[str, torch.device]) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**tokens)
        emb = getattr(outputs, "text_embeds", None)
        if emb is None:
            emb = outputs.pooler_output
        emb = emb.to(dtype=torch.float32)
        if self.normalize:
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return emb.to(device)

    def get_lang_emb(self, lang_strings: Union[str, Sequence[str]]) -> torch.Tensor:
        """Embed one or more language strings into a ``(B, embedding_dim)`` tensor."""
        if isinstance(lang_strings, str):
            lang_strings = [lang_strings]
        return self._embed(list(lang_strings), self.device)

    def encode(
        self,
        sentences: Union[str, Sequence[str]],
        batch_size: int = 32,
        convert_to_tensor: bool = False,
        convert_to_numpy: bool = True,
        device: Union[str, torch.device, None] = None,
        normalize_embeddings: Optional[bool] = None,
        **kwargs: Any,
    ):
        """``SentenceTransformer.encode``-compatible entry point.

        A single string returns a 1-D embedding; a sequence returns ``(B, embedding_dim)``.
        Extra keyword arguments are accepted and ignored for signature compatibility.
        """
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        target_device = device if device is not None else self.device

        previous_normalize = self.normalize
        if normalize_embeddings is not None:
            self.normalize = bool(normalize_embeddings)
        try:
            chunks = [
                self._embed(texts[i : i + batch_size], target_device) for i in range(0, len(texts), batch_size)
            ]
        finally:
            self.normalize = previous_normalize

        embeddings = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        if single:
            embeddings = embeddings[0]
        if convert_to_tensor:
            return embeddings
        if convert_to_numpy:
            return embeddings.cpu().numpy()
        return embeddings


def build_language_encoder(
    lang_encoder_type: str = "minilm",
    model_name: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
    **kwargs: Any,
):
    """Construct a language encoder.

    Args:
        lang_encoder_type: ``"minilm"`` / ``"sentence_transformer"`` (any sentence-transformers
            model id) or ``"clip"`` (CLIP text tower).
        model_name: HF model id; defaults to MiniLM-L6 / CLIP ViT-B/32 for the respective type.
        device: device the encoder runs on.
        **kwargs: forwarded to the encoder (e.g. ``use_projection``/``normalize`` for CLIP).
    """
    etype = (lang_encoder_type or "").lower()
    if etype in ("minilm", "sentence_transformer", "sentence-transformer"):
        return MiniLMLangEncoder(device=device, model_name=model_name or MINILM_DEFAULT_MODEL, **kwargs)
    if etype == "clip":
        return CLIPLangEncoder(device=device, model_name=model_name or CLIP_DEFAULT_MODEL, **kwargs)
    raise ValueError(
        f"Unknown lang_encoder_type: {lang_encoder_type!r} (expected one of {'|'.join(LANGUAGE_ENCODER_TYPES)})"
    )


def language_embedding_dim(encoder: Any, default: int = 384) -> int:
    """Best-effort embedding size of a language encoder (falls back to ``default``)."""
    dim = getattr(encoder, "embedding_dim", None)
    if dim is None and hasattr(encoder, "get_sentence_embedding_dimension"):
        dim = encoder.get_sentence_embedding_dimension()
    return int(dim) if dim else int(default)
