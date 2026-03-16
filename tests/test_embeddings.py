"""Tests for BiomedCLIP embedding module."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image


@pytest.fixture
def mock_open_clip():
    """Mock open_clip to avoid downloading model in tests."""
    mock_model = MagicMock()
    # encode_text returns (batch, 512) tensor
    mock_model.encode_text.side_effect = lambda x: _fake_encode(x)
    mock_model.encode_image.side_effect = lambda x: _fake_encode(x)
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = None

    mock_preprocess = lambda img: __import__("torch").randn(3, 224, 224)
    mock_tokenizer = MagicMock(
        side_effect=lambda texts, context_length=256: __import__("torch").randint(
            0, 1000, (len(texts), context_length)
        )
    )

    with patch("open_clip.create_model_and_transforms") as mock_create, \
         patch("open_clip.get_tokenizer") as mock_get_tok:
        mock_create.return_value = (mock_model, mock_preprocess, mock_preprocess)
        mock_get_tok.return_value = mock_tokenizer
        yield mock_model


def _fake_encode(x):
    """Create fake normalized embeddings."""
    import torch
    batch_size = x.shape[0]
    emb = torch.randn(batch_size, 512)
    return emb


@pytest.fixture
def embedder(mock_open_clip):
    """Create embedder with mocked model."""
    from src.rag.embeddings import BiomedCLIPEmbedder
    return BiomedCLIPEmbedder(device="cpu")


def test_text_embedding_shape(embedder):
    """Verify text embedding produces (512,) output."""
    embedding = embedder.embed_text("brain metastasis MRI segmentation")
    assert embedding.shape == (512,)


def test_text_embedding_normalized(embedder):
    """Verify text embedding is L2-normalized."""
    embedding = embedder.embed_text("brain metastasis treatment")
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 0.05, f"L2 norm should be ~1.0, got {norm}"


def test_batch_embedding(embedder):
    """Verify batch text embedding produces (N, 512) output."""
    texts = [
        "brain metastasis MRI",
        "stereotactic radiosurgery",
        "immunotherapy response",
    ]
    embeddings = embedder.embed_texts(texts)
    assert embeddings.shape == (3, 512)


def test_volume_embedding(embedder):
    """Verify MRI volume embedding produces (512,) output."""
    volume = np.random.randn(64, 64, 32).astype(np.float32)
    embedding = embedder.embed_mri_volume(volume, slice_axis=2, num_slices=8)
    assert embedding.shape == (512,)


def test_volume_embedding_normalized(embedder):
    """Verify volume embedding is L2-normalized."""
    volume = np.random.randn(64, 64, 32).astype(np.float32)
    embedding = embedder.embed_mri_volume(volume)
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 0.05, f"L2 norm should be ~1.0, got {norm}"


def test_cross_modal_similarity(embedder):
    """Verify cosine similarity between text and image is in valid range."""
    text_emb = embedder.embed_text("brain metastasis")
    volume = np.random.randn(64, 64, 32).astype(np.float32)
    img_emb = embedder.embed_mri_volume(volume)
    similarity = np.dot(text_emb, img_emb)
    assert -1.0 <= similarity <= 1.0, f"Cosine similarity out of range: {similarity}"


def test_image_embedding(embedder):
    """Verify PIL image embedding produces correct shape."""
    images = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
              for _ in range(4)]
    embeddings = embedder.embed_images(images)
    assert embeddings.shape == (4, 512)
