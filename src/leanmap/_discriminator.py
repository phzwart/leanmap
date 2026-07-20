"""Conformalized discriminator: a calibrated 'could this be real?' score.

A generative decoder produces a *smooth, lower-rank* approximation of the real
data manifold, so its samples are not exchangeable with real data -- a flexible
classifier can always separate them (AUC > 0.5). Rather than pretend otherwise,
we turn that classifier into a *calibrated one-sided test*: given a held-out set
of genuine real examples, a conformal p-value answers "what fraction of real
examples look at least this fake?" -- valid under exchangeability of the real
calibration and real test points, and correctly powerful against generated ones.

Mondrian stratification splits the calibration set by **leanmap region** (a cell
of the embedded space) so the false-rejection guarantee holds *locally*: a query
is judged against real calibration points from its own region of the manifold.
Sparse regions (< ``min_regional_pool`` calibration points) fall back to the
global pool so validity is preserved everywhere.

``rejection_sample`` uses the test as a quality gate: draw from a generator, keep
only samples whose conformal p-value clears ``alpha``. Kept samples then have a
calibrated false-keep rate, trading yield for fidelity with a guarantee.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ._train import resolve_device


class _DiscNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class LeanmapDiscriminator:
    """Real-vs-generated classifier + Mondrian conformal calibration over leanmap.

    Fit on real training examples and generated negatives, plus a *held-out* set
    of real examples for calibration (never shown to the classifier). ``p_value``
    returns a conformal 'could be real' p-value (>= alpha => not rejected).
    """

    def __init__(
        self,
        n_regions: int = 4,
        min_regional_pool: int = 10,
        input_scale: float = 16.0,
        hidden: int = 128,
        device: str | None = None,
    ) -> None:
        self.n_regions = int(n_regions)          # NB x NB grid over leanmap
        self.min_regional_pool = int(min_regional_pool)
        self.input_scale = float(input_scale)
        self.hidden = int(hidden)
        self.device = resolve_device(device)
        self.disc: _DiscNet | None = None

    # -- region binning over the embedded (leanmap) space ------------------
    def _fit_regions(self, Z: np.ndarray) -> None:
        self._lo = Z.min(0)
        self._hi = Z.max(0)

    def region_of(self, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(np.asarray(Z, np.float32))
        NB = self.n_regions
        span = (self._hi - self._lo) + 1e-9
        ix = np.clip(((Z[:, 0] - self._lo[0]) / span[0] * NB).astype(int), 0, NB - 1)
        iy = np.clip(((Z[:, 1] - self._lo[1]) / span[1] * NB).astype(int), 0, NB - 1)
        return ix * NB + iy

    # -- training ----------------------------------------------------------
    def fit(
        self,
        X_real: np.ndarray,
        X_generated: np.ndarray,
        X_calib: np.ndarray,
        Z_calib: np.ndarray,
        epochs: int = 300,
        batch_size: int = 256,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ) -> "LeanmapDiscriminator":
        torch.manual_seed(seed)
        np.random.seed(seed)
        s = self.input_scale
        Xr = np.asarray(X_real, np.float32) / s
        Xg = np.asarray(X_generated, np.float32) / s
        X = np.vstack([Xr, Xg]).astype(np.float32)
        y = np.concatenate([np.ones(len(Xr)), np.zeros(len(Xg))]).astype(np.float32)
        Xt = torch.as_tensor(X, device=self.device)
        yt = torch.as_tensor(y, device=self.device)

        self.disc = _DiscNet(X.shape[1], self.hidden).to(self.device)
        opt = torch.optim.Adam(self.disc.parameters(), lr=lr, weight_decay=weight_decay)
        bce = nn.BCEWithLogitsLoss()
        n = len(Xt)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                opt.zero_grad()
                bce(self.disc(Xt[idx]), yt[idx]).backward()
                opt.step()
        self.disc.eval()

        # calibration: nonconformity scores of held-out reals, binned by region
        self._fit_regions(np.asarray(Z_calib, np.float32))
        self.cal_scores_ = self.score(X_calib)
        self.cal_regions_ = self.region_of(Z_calib)
        return self

    # -- scoring -----------------------------------------------------------
    def score(self, X: np.ndarray) -> np.ndarray:
        """Nonconformity score = 1 - P(real); higher = more anomalous."""
        Xt = torch.as_tensor(np.asarray(X, np.float32) / self.input_scale, device=self.device)
        with torch.no_grad():
            return (1.0 - torch.sigmoid(self.disc(Xt))).cpu().numpy()

    def p_value(self, X: np.ndarray, Z: np.ndarray | None = None, mondrian: bool = True) -> np.ndarray:
        """Conformal p-value: P[a real example looks at least this fake].

        ``mondrian=True`` calibrates within the query's leanmap region (needs Z);
        regions with < ``min_regional_pool`` calibration points fall back to global.
        """
        s = self.score(X)
        if not mondrian:
            cal = self.cal_scores_
            return np.array([((cal >= v).sum() + 1) / (len(cal) + 1) for v in s])
        if Z is None:
            raise ValueError("mondrian=True requires embedding coordinates Z.")
        regions = self.region_of(Z)
        p = np.empty(len(s))
        for i, (v, r) in enumerate(zip(s, regions)):
            pool = self.cal_scores_[self.cal_regions_ == r]
            if len(pool) < self.min_regional_pool:
                pool = self.cal_scores_
            p[i] = ((pool >= v).sum() + 1) / (len(pool) + 1)
        return p

    def could_be_real(self, X: np.ndarray, Z: np.ndarray | None = None, alpha: float = 0.1,
                      mondrian: bool = True) -> np.ndarray:
        """Boolean gate: True if not rejected at level ``alpha`` (p >= alpha)."""
        return self.p_value(X, Z, mondrian=mondrian) >= alpha

    # -- rejection sampling as a quality gate ------------------------------
    def rejection_sample(self, generate, z_star, n_want: int = 8, alpha: float = 0.1,
                        mondrian: bool = True, batch: int = 256, max_draw: int = 20000,
                        seed: int = 0):
        """Keep only generated samples that pass the test.

        ``generate(z_batch, seed)`` must return an (n, n_features) array of draws
        for the given embedding coordinates. Returns (kept_x, kept_p, n_drawn).
        Yield ~= accept rate; getting ``n_want`` kept costs ~ n_want / accept draws.
        """
        rng = np.random.default_rng(seed)
        z_star = np.atleast_2d(np.asarray(z_star, np.float32))
        kept, kept_p, drawn = [], [], 0
        while len(kept) < n_want and drawn < max_draw:
            zb = np.repeat(z_star, batch, axis=0)
            x = np.asarray(generate(zb, int(rng.integers(1_000_000_000))), np.float32)
            zrep = np.repeat(z_star, batch, axis=0) if len(z_star) == 1 else zb
            p = self.p_value(x, zrep, mondrian=mondrian)
            drawn += len(x)
            for img, pp in zip(x[p >= alpha], p[p >= alpha]):
                kept.append(img)
                kept_p.append(pp)
                if len(kept) >= n_want:
                    break
        return np.array(kept[:n_want]), np.array(kept_p[:n_want]), drawn

    def accept_rate(self, generate, z_star, alpha: float = 0.1, mondrian: bool = True,
                   n: int = 1000, seed: int = 5) -> float:
        """Honest accept rate on a fixed pool of ``n`` draws (no early stop)."""
        z_star = np.atleast_2d(np.asarray(z_star, np.float32))
        zb = np.repeat(z_star, n, axis=0)
        x = np.asarray(generate(zb, seed), np.float32)
        p = self.p_value(x, zb, mondrian=mondrian)
        return float((p >= alpha).mean())

    # -- persistence -------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "n_regions": self.n_regions,
            "min_regional_pool": self.min_regional_pool,
            "input_scale": self.input_scale,
            "hidden": self.hidden,
            "in_dim": self.disc.net[0].in_features,
            "disc": {k: v.cpu().numpy() for k, v in self.disc.state_dict().items()},
            "cal_scores": self.cal_scores_,
            "cal_regions": self.cal_regions_,
            "lo": self._lo,
            "hi": self._hi,
        }

    @classmethod
    def load_state(cls, sd: dict, device: str | None = None) -> "LeanmapDiscriminator":
        obj = cls(
            n_regions=sd["n_regions"],
            min_regional_pool=sd["min_regional_pool"],
            input_scale=sd["input_scale"],
            hidden=sd["hidden"],
            device=device,
        )
        obj.disc = _DiscNet(int(sd["in_dim"]), sd["hidden"]).to(obj.device)
        obj.disc.load_state_dict({k: torch.as_tensor(v) for k, v in sd["disc"].items()})
        obj.disc.eval()
        obj.cal_scores_ = np.asarray(sd["cal_scores"], np.float32)
        obj.cal_regions_ = np.asarray(sd["cal_regions"])
        obj._lo = np.asarray(sd["lo"], np.float32)
        obj._hi = np.asarray(sd["hi"], np.float32)
        return obj
