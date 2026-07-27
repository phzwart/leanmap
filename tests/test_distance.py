"""Stage-1 tests: distance protocol and chunked_cdist."""

from __future__ import annotations

import numpy as np
import torch

from leanmap.distance import (
    CallableDistance,
    EuclideanDistance,
    chunked_cdist,
    is_differentiable,
)


def test_chunked_cdist_matches_cdist():
    torch.manual_seed(0)
    A = torch.randn(200, 16)
    B = torch.randn(300, 16)
    dist = EuclideanDistance()
    full = torch.cdist(A, B, p=2)
    for ca, cb in [(32, 64), (50, 100), (200, 300), (7, 13)]:
        out = chunked_cdist(dist, A, B, chunk_a=ca, chunk_b=cb, out_device="cpu")
        assert isinstance(out, torch.Tensor)
        assert torch.allclose(out, full.cpu(), atol=1e-5, rtol=1e-5)


def test_chunked_cdist_topk_matches_full():
    torch.manual_seed(1)
    A = torch.randn(80, 8)
    B = torch.randn(120, 8)
    dist = EuclideanDistance()
    full = chunked_cdist(dist, A, B, chunk_a=16, chunk_b=32, out_device="cpu")
    assert isinstance(full, torch.Tensor)
    k = 7
    vals, idx = chunked_cdist(
        dist, A, B, chunk_a=16, chunk_b=32, out_device="cpu", topk=k
    )
    ref_vals, ref_idx = torch.topk(full, k=k, dim=1, largest=False)
    assert torch.allclose(vals, ref_vals, atol=1e-5)
    # Indices of equal distances may differ; check values at returned indices
    gathered = full.gather(1, idx)
    assert torch.allclose(gathered, ref_vals, atol=1e-5)


def test_is_differentiable_euclidean_true():
    assert is_differentiable(EuclideanDistance(), D=8, device="cpu") is True


def test_is_differentiable_numpy_roundtrip_false():
    def numpy_roundtrip(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        a = A.detach().cpu().numpy()
        b = B.detach().cpu().numpy()
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        return torch.as_tensor(d, dtype=torch.float32, device=A.device)

    assert is_differentiable(CallableDistance(numpy_roundtrip), D=8, device="cpu") is False
