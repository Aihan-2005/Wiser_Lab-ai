from __future__ import annotations

import torch
import torch.nn.functional as F

from persian_sea.semantic.candidates import interval_iou


def mil_bag_score(
    text_vector: torch.Tensor,
    span_vectors: torch.Tensor,
    mil_temperature: float,
) -> torch.Tensor:
    similarities = span_vectors @ text_vector
    tau = max(float(mil_temperature), 1e-4)
    return tau * torch.logsumexp(similarities / tau, dim=0)


def contrastive_bag_loss(
    text_vectors: torch.Tensor,
    sample_outputs: list,
    video_ids: list[str],
    mil_temperature: float,
    contrastive_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Text i should retrieve visual bag i.

    Off-diagonal samples from the same source video are masked rather than
    treated as hard negatives, reducing false negatives among neighbouring
    subtitles in one episode.
    """
    B = text_vectors.shape[0]
    matrix = text_vectors.new_empty((B, B))

    for i in range(B):
        for j in range(B):
            matrix[i, j] = mil_bag_score(
                text_vectors[i],
                sample_outputs[j].vectors,
                mil_temperature,
            )

    # Mask same-video off-diagonal entries.
    for i in range(B):
        for j in range(B):
            if i != j and video_ids[i] == video_ids[j]:
                matrix[i, j] = -1e4

    logits = matrix / max(float(contrastive_temperature), 1e-4)
    target = torch.arange(B, device=logits.device)

    loss_t2v = F.cross_entropy(logits, target)
    loss_v2t = F.cross_entropy(logits.T, target)

    return 0.5 * (loss_t2v + loss_v2t), matrix.detach()


def supervised_span_loss(
    sample_outputs: list,
    gt_start: torch.Tensor,
    gt_end: torch.Tensor,
    iou_temperature: float,
) -> tuple[torch.Tensor, int]:
    losses = []
    supervised_count = 0
    tau = max(float(iou_temperature), 1e-4)

    for i, output in enumerate(sample_outputs):
        if not (
            torch.isfinite(gt_start[i])
            and torch.isfinite(gt_end[i])
        ):
            continue

        gs = float(gt_start[i].item())
        ge = float(gt_end[i].item())

        ious = torch.tensor(
            [
                interval_iou(
                    span.start_time,
                    span.end_time,
                    gs,
                    ge,
                )
                for span in output.spans
            ],
            device=output.final_scores.device,
            dtype=output.final_scores.dtype,
        )

        if float(ious.max().item()) <= 0.0:
            # If segmentation/candidate search misses the GT, supervise the
            # closest span midpoint rather than creating a zero target.
            gt_mid = (gs + ge) / 2.0
            mids = torch.tensor(
                [span.midpoint for span in output.spans],
                device=ious.device,
                dtype=ious.dtype,
            )
            target = torch.zeros_like(ious)
            target[torch.argmin(torch.abs(mids - gt_mid))] = 1.0
        else:
            target = torch.softmax(ious / tau, dim=0)

        log_prob = torch.log_softmax(output.final_scores, dim=0)
        losses.append(-(target * log_prob).sum())
        supervised_count += 1

    if not losses:
        zero = gt_start.new_tensor(0.0)
        return zero, 0

    return torch.stack(losses).mean(), supervised_count
