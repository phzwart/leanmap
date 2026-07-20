"""Training loop for the parametric mapper."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._graph import FuzzyGraphData
from ._model import AttentionMapper, DeployableMapper, ParametricMapper, pca_components


def resolve_device(device: str | None = None) -> torch.device:
    """Pick the best available torch device: CUDA -> MPS (Apple GPU) -> CPU.

    Pass an explicit string (e.g. "cpu", "cuda", "mps") to override. On Apple
    Silicon, "mps" uses the Metal GPU; it is selected automatically when
    available and CUDA is not.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def umap_pair_nll(
    squared_distance: torch.Tensor,
    *,
    a: float,
    b: float,
    positive: bool,
) -> torch.Tensor:
    """Cross-entropy between graph membership and the 1/(1+a d^2b) kernel."""
    log_t = math.log(a) + b * torch.log(squared_distance.clamp_min(1e-12))
    return F.softplus(log_t if positive else -log_t)


def mid_weight_schedule(
    epoch: int,
    epochs: int,
    start: float,
    end: float,
) -> float:
    """Anneal the ranking-loss weight: flat, then linear ramp, then flat."""
    progress = epoch / max(1, epochs - 1)
    if progress < 0.20:
        return start
    if progress < 0.80:
        fraction = (progress - 0.20) / 0.60
        return start + fraction * (end - start)
    return end


def train_parametric_mapper(
    graph: FuzzyGraphData,
    *,
    epochs: int = 25,
    batch_size: int = 4096,
    pairs_per_epoch: int | None = None,
    negative_sample_rate: int = 5,
    candidate_count: int = 6,
    mid_weight_start: float = 1.0,
    mid_weight_end: float = 0.05,
    rank_margin: float = 0.05,
    rank_temperature: float = 0.20,
    repulsion_strength: float = 1.0,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden_dims: tuple[int, ...] = (64, 64),
    n_components: int = 2,
    seed: int = 42,
    device: str | None = None,
    verbose: bool = True,
    init_encoder_state: dict | None = None,
) -> DeployableMapper:
    """Train the PCA-anchored encoder to embed the fuzzy graph.

    Loss = attractive (graph edges) + repulsion_strength * repulsive
    (negatives) + mid_weight * ranking (triplet ordering), with the ranking
    term annealed from mid_weight_start to mid_weight_end.
    """
    if candidate_count < 3:
        raise ValueError("candidate_count must be at least 3")
    if negative_sample_rate < 1:
        raise ValueError("negative_sample_rate must be at least 1")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_obj = resolve_device(device)

    x_scaled = graph.x_scaled
    n, input_dim = x_scaled.shape
    components = pca_components(x_scaled, n_components=n_components)
    encoder = ParametricMapper(input_dim, components, hidden_dims).to(device_obj)
    if init_encoder_state is not None:
        # Warm-start: load pre-trained encoder weights (e.g. from a regression
        # pre-fit onto reference coordinates) before graph-loss fine-tuning.
        encoder.load_state_dict(init_encoder_state)

    x_tensor = torch.from_numpy(x_scaled).to(device_obj)
    edge_head = torch.from_numpy(graph.head)
    edge_tail = torch.from_numpy(graph.tail)
    edge_weight = torch.from_numpy(graph.weight)
    edge_count = edge_head.numel()
    if edge_count == 0:
        raise ValueError("The fuzzy graph has no edges")

    mean_edge_weight = float(graph.weight.mean())
    pairs_per_epoch = pairs_per_epoch or 4 * n
    steps_per_epoch = math.ceil(pairs_per_epoch / batch_size)

    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=learning_rate * 0.05,
    )

    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)

    for epoch in range(epochs):
        encoder.train()
        mid_weight = mid_weight_schedule(
            epoch,
            epochs,
            mid_weight_start,
            mid_weight_end,
        )

        running = np.zeros(4, dtype=np.float64)

        for _ in range(steps_per_epoch):
            edge_ids = torch.randint(
                edge_count,
                (batch_size,),
                generator=cpu_generator,
            )
            anchor = edge_head[edge_ids].to(device_obj, dtype=torch.long)
            near = edge_tail[edge_ids].to(device_obj, dtype=torch.long)
            event_weight = edge_weight[edge_ids].to(device_obj) / mean_edge_weight

            candidates = torch.randint(
                n,
                (batch_size, candidate_count),
                device=device_obj,
            )
            candidates = torch.where(
                candidates == anchor[:, None],
                (candidates + 1) % n,
                candidates,
            )

            high_d2 = (
                x_tensor[candidates] - x_tensor[anchor, None, :]
            ).square().sum(dim=-1)
            ordered = high_d2.argsort(dim=1)
            mid = candidates.gather(1, ordered[:, 1:2]).squeeze(1)
            rank_far = candidates.gather(1, ordered[:, -1:]).squeeze(1)

            negative = torch.randint(
                n,
                (batch_size, negative_sample_rate),
                device=device_obj,
            )
            negative = torch.where(
                negative == anchor[:, None],
                (negative + 1) % n,
                negative,
            )

            all_indices = torch.cat(
                (anchor, near, mid, rank_far, negative.reshape(-1))
            )
            all_z = encoder(x_tensor[all_indices])

            B = batch_size
            z_anchor = all_z[0:B]
            z_near = all_z[B : 2 * B]
            z_mid = all_z[2 * B : 3 * B]
            z_rank_far = all_z[3 * B : 4 * B]
            z_negative = all_z[4 * B :].reshape(
                B, negative_sample_rate, encoder.out_dim
            )

            near_d2 = (z_anchor - z_near).square().sum(dim=-1)
            mid_d2 = (z_anchor - z_mid).square().sum(dim=-1)
            rank_far_d2 = (z_anchor - z_rank_far).square().sum(dim=-1)
            negative_d2 = (
                z_anchor[:, None, :] - z_negative
            ).square().sum(dim=-1)

            positive_loss = (
                event_weight
                * umap_pair_nll(
                    near_d2,
                    a=graph.a,
                    b=graph.b,
                    positive=True,
                )
            ).mean()

            negative_nll = umap_pair_nll(
                negative_d2,
                a=graph.a,
                b=graph.b,
                positive=False,
            )
            negative_loss = (
                event_weight[:, None] * negative_nll
            ).sum(dim=1).mean()

            rank_near = 0.5 * torch.log1p(near_d2)
            rank_mid = 0.5 * torch.log1p(mid_d2)
            rank_far_value = 0.5 * torch.log1p(rank_far_d2)

            rank_loss = (
                rank_temperature
                * (
                    F.softplus(
                        (rank_near - rank_mid + rank_margin)
                        / rank_temperature
                    )
                    + F.softplus(
                        (rank_mid - rank_far_value + rank_margin)
                        / rank_temperature
                    )
                )
            ).mean()

            loss = (
                positive_loss
                + repulsion_strength * negative_loss
                + mid_weight * rank_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            optimizer.step()

            running += np.array(
                [
                    loss.item(),
                    positive_loss.item(),
                    negative_loss.item(),
                    rank_loss.item(),
                ]
            )

        scheduler.step()
        running /= steps_per_epoch
        if verbose:
            print(
                f"epoch={epoch + 1:03d} total={running[0]:.4f} "
                f"near={running[1]:.4f} negative={running[2]:.4f} "
                f"rank={running[3]:.4f} mid_weight={mid_weight:.3f}"
            )

    return DeployableMapper(encoder, graph.mean, graph.scale).to(device_obj).eval()


def _prepare_order_constraints(
    constraints: list[dict] | None, n_samples: int, seed: int
) -> list[dict] | None:
    """Validate + precompute tensors for supervised axis-ordering constraints.

    Each constraint dict:
      axis   : "x"/"y" (or 0/1)   -- which embedding coordinate to order along
      kind   : "ordinal" | "separate"
      labels : (n,) array of category / group ids per training sample
      order  : optional list of ids giving the desired low->high order along
               the axis (default: sorted unique ids)
      weight : float (default 1.0)
      margin : float, minimum gap between consecutive centroids in std units
               (default 0.4 for ordinal, 0.6 for separate)

    Returns a list of prepared dicts with an ``axis`` int, an int64 tensor
    ``group_of`` remapping each sample to its position in ``order`` (0..K-1),
    ``k`` groups, ``weight``, ``margin`` -- or None if no constraints.
    """
    if not constraints:
        return None
    prepared = []
    for con in constraints:
        axis = con.get("axis", "x")
        axis_i = {"x": 0, "y": 1, 0: 0, 1: 1}[axis]
        kind = con.get("kind", "ordinal")
        labels = np.asarray(con["labels"])
        if labels.shape[0] != n_samples:
            raise ValueError("order constraint labels must be length n_samples")
        order = con.get("order")
        if order is None:
            order = sorted(np.unique(labels).tolist())
        id_to_rank = {gid: r for r, gid in enumerate(order)}
        try:
            group_of = np.array([id_to_rank[v] for v in labels], dtype=np.int64)
        except KeyError as exc:  # a label not covered by `order`
            raise ValueError(
                f"order constraint 'order' is missing label {exc}"
            ) from None
        margin = con.get("margin", 0.4 if kind == "ordinal" else 0.6)
        prepared.append(
            dict(
                axis=axis_i,
                group_of=torch.from_numpy(group_of),
                k=len(order),
                weight=float(con.get("weight", 1.0)),
                margin=float(margin),
            )
        )
    return prepared


def _axis_order_loss(
    emb: torch.Tensor, prepared: list[dict]
) -> torch.Tensor:
    """Sum of chain-ranking penalties pushing group centroids into order.

    ``emb`` is the (m, 2) embedding of the ordering subsample; each prepared
    constraint's ``group_of`` must be aligned to the same subsample rows.
    """
    total = emb.new_zeros(())
    for con in prepared:
        coord = emb[:, con["axis"]]
        onehot = torch.nn.functional.one_hot(con["group_of"], con["k"]).to(coord.dtype)
        counts = onehot.sum(0).clamp_min(1e-6)
        means = (onehot * coord[:, None]).sum(0) / counts   # (k,) in order 0..K-1
        std = coord.std().clamp_min(1e-6)
        gaps = (means[1:] - means[:-1]) / std
        total = total + con["weight"] * torch.nn.functional.softplus(
            con["margin"] - gaps
        ).mean()
    return total


def train_attention_mapper(
    graph: FuzzyGraphData,
    landmark_hd: np.ndarray,
    landmark_emb: np.ndarray,
    *,
    epochs: int = 120,
    batch_size: int = 512,
    pairs_per_epoch: int | None = None,
    negative_sample_rate: int = 10,
    repulsion_strength: float = 1.0,
    learning_rate: float = 5e-3,
    weight_decay: float = 1e-5,
    hidden_dims: tuple[int, ...] = (128, 128),
    attn_dim: int = 64,
    heads: int = 4,
    attn_layers: int = 2,
    learn_landmarks: bool = True,
    landmark_lr_mult: float = 1.0,
    gram_anchor_weight: float = 1.0,
    distance_kernel: str = "linear",
    attend_top_p: int | None = None,
    order_constraints: list[dict] | None = None,
    order_subsample: int = 2048,
    seed: int = 42,
    device: str | None = None,
    verbose: bool = True,
    on_epoch=None,
) -> DeployableMapper:
    """Train the landmark-attention encoder with the UMAP attract/repel loss.

    ``landmark_hd`` must be in the same standardized space as ``graph.x_scaled``;
    ``landmark_emb`` are the landmarks' reference 2D coordinates. Returns a
    DeployableMapper (accepts raw input) wrapping the trained AttentionMapper.

    learn_landmarks : if True (default), the high-D landmark positions are free
        parameters optimized jointly with the network (Titsias-style inducing
        inputs), initialized at the supplied data-anchored positions.
    landmark_lr_mult : learning-rate multiplier for the landmark parameter
        relative to the base rate (e.g. 0.3 to move them gently).
    gram_anchor_weight : weight on the Gram-anchor penalty that keeps the
        learnable landmark cloud's relative geometry near its initialization,
        preventing runoff while allowing rigid motion. Set 0 to disable.
    order_constraints : optional supervised axis-ordering constraints; see
        ``_prepare_order_constraints``. Adds a per-axis chain-ranking loss so a
        label gradient is laid out along X and/or Y.
    order_subsample : number of training points used to estimate group centroids
        for the ordering loss each step (all points if n is smaller).
    """
    if negative_sample_rate < 1:
        raise ValueError("negative_sample_rate must be at least 1")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_obj = resolve_device(device)
    x_scaled = graph.x_scaled
    n, input_dim = x_scaled.shape
    # output dim follows the reference-coordinate dimensionality of the landmarks
    n_components = int(np.asarray(landmark_emb).shape[1])
    components = pca_components(x_scaled, n_components=n_components)
    encoder = AttentionMapper(
        input_dim,
        components,
        landmark_hd,
        landmark_emb,
        hidden_dims=hidden_dims,
        attn_dim=attn_dim,
        heads=heads,
        attn_layers=attn_layers,
        learn_landmarks=learn_landmarks,
        distance_kernel=distance_kernel,
        attend_top_p=attend_top_p,
    ).to(device_obj)
    prepared_order = _prepare_order_constraints(order_constraints, n, seed)

    x_tensor = torch.from_numpy(x_scaled).to(device_obj)
    edge_head = torch.from_numpy(graph.head)
    edge_tail = torch.from_numpy(graph.tail)
    edge_weight = torch.from_numpy(graph.weight)
    edge_count = edge_head.numel()
    if edge_count == 0:
        raise ValueError("The fuzzy graph has no edges")
    mean_edge_weight = float(graph.weight.mean())
    pairs_per_epoch = pairs_per_epoch or 8 * n
    steps_per_epoch = math.ceil(pairs_per_epoch / batch_size)

    if learn_landmarks and landmark_lr_mult != 1.0:
        landmark_params = [encoder.landmark_hd]
        other_params = [
            p for name, p in encoder.named_parameters() if name != "landmark_hd"
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": learning_rate},
                {"params": landmark_params, "lr": learning_rate * landmark_lr_mult},
            ],
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            encoder.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=learning_rate * 0.05
    )
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)

    for epoch in range(epochs):
        encoder.train()
        running = 0.0
        for _ in range(steps_per_epoch):
            edge_ids = torch.randint(
                edge_count, (batch_size,), generator=cpu_generator
            )
            anchor = edge_head[edge_ids].to(device_obj, dtype=torch.long)
            near = edge_tail[edge_ids].to(device_obj, dtype=torch.long)
            weight = edge_weight[edge_ids].to(device_obj) / mean_edge_weight
            negative = torch.randint(
                n, (batch_size, negative_sample_rate), device=device_obj
            )
            negative = torch.where(
                negative == anchor[:, None], (negative + 1) % n, negative
            )
            batch_idx = torch.cat((anchor, near, negative.reshape(-1)))
            emb = encoder(x_tensor[batch_idx])
            B = batch_size
            z_anchor = emb[:B]
            z_near = emb[B : 2 * B]
            z_negative = emb[2 * B :].reshape(
                B, negative_sample_rate, encoder.out_dim
            )
            near_d2 = (z_anchor - z_near).square().sum(-1)
            negative_d2 = (z_anchor[:, None, :] - z_negative).square().sum(-1)
            positive_loss = (
                weight * umap_pair_nll(near_d2, a=graph.a, b=graph.b, positive=True)
            ).mean()
            negative_loss = (
                weight[:, None]
                * umap_pair_nll(negative_d2, a=graph.a, b=graph.b, positive=False)
            ).sum(dim=1).mean()
            loss = positive_loss + repulsion_strength * negative_loss
            if gram_anchor_weight > 0.0 and learn_landmarks:
                loss = loss + gram_anchor_weight * encoder.gram_penalty()
            if prepared_order is not None:
                if n > order_subsample:
                    sub = torch.randint(
                        n, (order_subsample,), generator=cpu_generator
                    ).to(device_obj)
                    sub_cpu = sub.cpu()
                else:
                    sub = torch.arange(n, device=device_obj)
                    sub_cpu = sub.cpu()
                emb_sub = encoder(x_tensor[sub])
                cons_view = [
                    {**con, "group_of": con["group_of"][sub_cpu].to(device_obj)}
                    for con in prepared_order
                ]
                loss = loss + _axis_order_loss(emb_sub, cons_view)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            optimizer.step()
            running += loss.item()
        scheduler.step()
        ep_loss = running / steps_per_epoch
        if verbose:
            print(f"epoch={epoch + 1:03d} loss={ep_loss:.4f}")
        if on_epoch is not None:
            # give the caller the full embedding + loss for snapshots/plots
            encoder.eval()
            with torch.no_grad():
                snap = encoder(x_tensor).cpu().numpy()
            encoder.train()
            on_epoch(epoch + 1, snap, ep_loss)

    return DeployableMapper(encoder, graph.mean, graph.scale).to(device_obj).eval()


@torch.inference_mode()
def transform(
    mapper: DeployableMapper,
    X_new: np.ndarray,
    *,
    batch_size: int = 65536,
) -> np.ndarray:
    """Embed new raw data through a trained mapper, in batches."""
    X_new = np.asarray(X_new, dtype=np.float32, order="C")
    if X_new.ndim != 2:
        raise ValueError("X_new must be a 2D array")

    if len(X_new) == 0:
        out_dim = int(getattr(mapper.encoder, "out_dim", 2))
        return np.empty((0, out_dim), dtype=np.float32)

    device = next(mapper.parameters()).device
    output: list[torch.Tensor] = []
    for start in range(0, len(X_new), batch_size):
        batch = torch.from_numpy(X_new[start : start + batch_size]).to(device)
        output.append(mapper(batch).cpu())
    return torch.cat(output, dim=0).numpy()
