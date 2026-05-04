"""Representation diagnostics for encoded multimodal feature batches."""

from __future__ import annotations

from typing import Any, Dict

import torch


class RepresentationLayer:
    """Compute lightweight statistics for modality embeddings."""

    @staticmethod
    def summarize(encoded_batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"modalities": {}, "pairwise_cosine": {}}
        tensors = {
            k: v.detach().float().cpu()
            for k, v in encoded_batch.items()
            if isinstance(v, torch.Tensor) and v.ndim == 2 and v.numel() > 0
        }

        for key, tensor in tensors.items():
            norms = torch.linalg.norm(tensor, dim=1)
            summary["modalities"][key] = {
                "shape": list(tensor.shape),
                "mean_norm": float(norms.mean().item()),
                "std_norm": float(norms.std(unbiased=False).item()),
                "feature_mean_abs": float(tensor.abs().mean().item()),
            }

        keys = list(tensors.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a = tensors[keys[i]]
                b = tensors[keys[j]]
                dim = min(a.shape[1], b.shape[1])
                if dim <= 0:
                    continue
                a_n = a[:, :dim]
                b_n = b[:, :dim]
                a_n = a_n / (a_n.norm(dim=1, keepdim=True) + 1e-8)
                b_n = b_n / (b_n.norm(dim=1, keepdim=True) + 1e-8)
                cosine = (a_n * b_n).sum(dim=1).mean().item()
                summary["pairwise_cosine"][f"{keys[i]}__{keys[j]}"] = float(cosine)

        return summary
