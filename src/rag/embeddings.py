"""
BiomedCLIP embedding module for brain metastasis RAG pipeline.
Provides unified text + image embeddings in a shared 512-dim latent space.
"""

import numpy as np
import torch
from pathlib import Path
from typing import List

import yaml
from PIL import Image


class BiomedCLIPEmbedder:
    """
    Unified text and image embedder using BiomedCLIP.
    Produces 512-dim vectors in a shared cross-modal space.
    """

    def __init__(self, device: str = "cuda", config_path: str = None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        emb_config = config.get("embedding", {})
        self.model_name = emb_config.get(
            "model_name",
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )
        self.context_length = emb_config.get("context_length", 256)
        self.embedding_dim = emb_config.get("embedding_dim", 512)
        self.batch_size = emb_config.get("batch_size", 32)

        # Load model
        import open_clip

        self.model, self.preprocess_train, self.preprocess_val = (
            open_clip.create_model_and_transforms(self.model_name)
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns (512,) vector."""
        tokens = self.tokenizer([text], context_length=self.context_length).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy()

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts. Returns (N, 512) array."""
        all_features = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            tokens = self.tokenizer(batch, context_length=self.context_length).to(
                self.device
            )
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())
        return np.concatenate(all_features, axis=0)

    @torch.no_grad()
    def embed_images(self, pil_images: List[Image.Image]) -> np.ndarray:
        """Embed a batch of PIL images. Returns (N, 512) array."""
        all_features = []
        for i in range(0, len(pil_images), self.batch_size):
            batch = pil_images[i : i + self.batch_size]
            tensors = torch.stack([self.preprocess_val(img) for img in batch]).to(
                self.device
            )
            features = self.model.encode_image(tensors)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())
        return np.concatenate(all_features, axis=0)

    @torch.no_grad()
    def embed_mri_volume(
        self,
        volume: np.ndarray,
        slice_axis: int = 2,
        num_slices: int = 16,
    ) -> np.ndarray:
        """
        Embed a 3D MRI volume by sampling slices and averaging vision embeddings.

        Args:
            volume: 3D array (H, W, D)
            slice_axis: Axis to slice along
            num_slices: Number of slices to sample

        Returns:
            (512,) averaged embedding vector
        """
        D = volume.shape[slice_axis]
        slice_indices = np.linspace(0, D - 1, num_slices, dtype=int)

        pil_images = []
        for idx in slice_indices:
            if slice_axis == 0:
                slice_2d = volume[idx, :, :]
            elif slice_axis == 1:
                slice_2d = volume[:, idx, :]
            else:
                slice_2d = volume[:, :, idx]

            # Normalize to 0-255 uint8
            s_min, s_max = slice_2d.min(), slice_2d.max()
            if s_max > s_min:
                normalized = ((slice_2d - s_min) / (s_max - s_min) * 255).astype(
                    np.uint8
                )
            else:
                normalized = np.zeros_like(slice_2d, dtype=np.uint8)

            # Convert grayscale to RGB PIL image
            pil_img = Image.fromarray(normalized, mode="L").convert("RGB")
            pil_images.append(pil_img)

        # Embed all slices
        embeddings = self.embed_images(pil_images)

        # Average across slices
        avg_embedding = embeddings.mean(axis=0)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)

        return avg_embedding
