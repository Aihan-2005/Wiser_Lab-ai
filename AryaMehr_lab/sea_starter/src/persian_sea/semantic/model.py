from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from persian_sea.models.visual_encoder import PoseRGBVisualEncoder
from persian_sea.semantic.candidates import (
    CandidateSpan,
    enumerate_candidate_spans,
)


class CachedTextProjector(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x.float()), dim=-1)


class CandidateContextEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        layers: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Relative position in candidate window + log duration.
        self.time_mlp = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=layers,
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        segment_embeddings: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, M, D = segment_embeddings.shape

        positions = torch.linspace(
            -1.0,
            1.0,
            M,
            device=segment_embeddings.device,
            dtype=segment_embeddings.dtype,
        ).view(1, M).expand(B, M)

        duration = (ends - starts).clamp_min(1e-3)
        log_duration = torch.log1p(duration)

        time_features = torch.stack(
            [positions, log_duration],
            dim=-1,
        )

        h = segment_embeddings + self.time_mlp(time_features)
        h = self.encoder(
            h,
            src_key_padding_mask=~mask,
        )
        h = self.final_norm(h)

        # Zero padded slots again.
        return h * mask.unsqueeze(-1).to(h.dtype)


@dataclass
class SampleSpans:
    spans: list[CandidateSpan]
    vectors: torch.Tensor
    semantic_scores: torch.Tensor
    final_scores: torch.Tensor


class SemanticAligner(nn.Module):
    def __init__(
        self,
        text_input_dim: int = 1024,
        rgb_input_dim: int = 768,
        num_views: int = 4,
        embedding_dim: int = 512,
    ):
        super().__init__()

        self.visual = PoseRGBVisualEncoder(
            pose_input_dim=4,
            rgb_input_dim=rgb_input_dim,
            model_dim=384,
            output_dim=embedding_dim,
            num_views=num_views,
        )

        self.text = CachedTextProjector(
            input_dim=text_input_dim,
            output_dim=embedding_dim,
        )

        self.context = CandidateContextEncoder(
            dim=embedding_dim,
        )

        self.span_projection = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def encode_segments(
        self,
        pose: torch.Tensor,
        rgb: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        pose: [B,M,K,75,4]
        rgb:  [B,M,K,V,768]
        """
        B, M = pose.shape[:2]
        device = pose.device

        valid = candidate_mask.view(-1)

        flat_pose = pose.reshape(B * M, *pose.shape[2:])
        flat_rgb = rgb.reshape(B * M, *rgb.shape[2:])

        output = torch.zeros(
            (B * M, self.text.net[-1].out_features),
            device=device,
            dtype=torch.float32,
        )

        if valid.any():
            encoded = self.visual(
                flat_pose[valid],
                flat_rgb[valid],
            )["fused"]
            output[valid] = encoded.float()

        return output.reshape(B, M, -1)

    def contextualize(
        self,
        segment_embeddings: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.context(
            segment_embeddings,
            starts,
            ends,
            mask,
        )

    def _span_vectors(
        self,
        contextual: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        valid_count: int,
        max_span_segments: int,
        max_span_seconds: float,
    ) -> tuple[list[CandidateSpan], torch.Tensor]:
        starts_np = starts[:valid_count].detach().cpu().numpy()
        ends_np = ends[:valid_count].detach().cpu().numpy()

        spans = enumerate_candidate_spans(
            starts_np,
            ends_np,
            max_span_segments=max_span_segments,
            max_span_seconds=max_span_seconds,
        )

        vectors = []
        for span in spans:
            pooled = contextual[
                span.start_index:span.end_index
            ].mean(dim=0)
            pooled = self.span_projection(pooled)
            vectors.append(F.normalize(pooled, dim=-1))

        if vectors:
            return spans, torch.stack(vectors, dim=0)

        # Defensive fallback: one segment.
        fallback = CandidateSpan(
            0,
            1,
            float(starts_np[0]),
            float(ends_np[0]),
        )
        vec = F.normalize(
            self.span_projection(contextual[0]),
            dim=-1,
        )
        return [fallback], vec.unsqueeze(0)

    def forward(
        self,
        batch: dict,
        *,
        max_span_segments: int,
        max_span_seconds: float,
        align_temperature: float,
        temporal_prior_weight: float,
        temporal_sigma_seconds: float,
        duration_prior_weight: float,
    ) -> dict:
        segment = self.encode_segments(
            batch["pose"],
            batch["rgb"],
            batch["candidate_mask"],
        )

        contextual = self.contextualize(
            segment,
            batch["segment_starts"],
            batch["segment_ends"],
            batch["candidate_mask"],
        )

        text_vec = self.text(batch["text_embedding"])

        sample_outputs: list[SampleSpans] = []

        for i in range(segment.shape[0]):
            valid_count = int(
                batch["candidate_mask"][i].sum().item()
            )

            spans, span_vectors = self._span_vectors(
                contextual[i],
                batch["segment_starts"][i],
                batch["segment_ends"][i],
                valid_count,
                max_span_segments,
                max_span_seconds,
            )

            semantic = (
                span_vectors @ text_vec[i]
            ) / align_temperature

            rough_start = batch["rough_start"][i]
            rough_end = batch["rough_end"][i]
            rough_mid = (rough_start + rough_end) / 2.0
            rough_duration = (rough_end - rough_start).clamp_min(1e-3)

            span_mid = torch.tensor(
                [s.midpoint for s in spans],
                device=semantic.device,
                dtype=semantic.dtype,
            )
            span_duration = torch.tensor(
                [s.duration for s in spans],
                device=semantic.device,
                dtype=semantic.dtype,
            ).clamp_min(1e-3)

            prior = -0.5 * (
                (span_mid - rough_mid)
                / max(temporal_sigma_seconds, 1e-3)
            ) ** 2

            prior = prior - duration_prior_weight * torch.abs(
                torch.log(span_duration / rough_duration)
            )

            final = semantic + temporal_prior_weight * prior

            sample_outputs.append(
                SampleSpans(
                    spans=spans,
                    vectors=span_vectors,
                    semantic_scores=semantic,
                    final_scores=final,
                )
            )

        return {
            "text_vectors": text_vec,
            "segment_vectors": segment,
            "contextual_vectors": contextual,
            "samples": sample_outputs,
        }
