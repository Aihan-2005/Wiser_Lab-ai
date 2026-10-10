from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class BGEM3DenseExtractor:
    """
    Frozen dense BGE-M3 encoder.

    BGE-M3's dense representation is the normalized hidden state of [CLS].
    We keep it frozen and cache text vectors offline, mirroring the frozen
    DINOv2 cache used for RGB.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "auto",
        amp: bool = True,
        max_length: int = 256,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.amp = amp and self.device.type == "cuda"
        self.max_length = int(max_length)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval().to(self.device)

        for p in self.model.parameters():
            p.requires_grad = False

        self.hidden_size = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.hidden_size), dtype=np.float32)

        chunks = []

        for start in range(0, len(texts), batch_size):
            batch_text = texts[start:start + batch_size]

            tokens = self.tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {
                k: v.to(self.device)
                for k, v in tokens.items()
            }

            context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                )
                if self.amp
                else nullcontext()
            )

            with context:
                output = self.model(**tokens)
                dense = output.last_hidden_state[:, 0]
                dense = F.normalize(dense.float(), dim=-1)

            chunks.append(dense.cpu().numpy())

        return np.concatenate(chunks, axis=0).astype(np.float32)
