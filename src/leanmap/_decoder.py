"""Generative decoder: sample p(x | z) instead of only the conditional mean.

A plain regression decoder ``D(z)`` learns ``E[x | z]`` -- the conditional mean
of the images that embed to ``z``. Because the encoder is lossy (high-D -> 2D or
d-D), that mean is a blurred, class-typical shape: all within-cell variation is
averaged away. To recover sharp, *varied* reconstructions we model the full
conditional distribution and draw from it:

    x = D(z) + V @ c,      c ~ flow( . | z )

where ``D`` is the mean decoder, ``V`` is a PCA basis of the *residuals*
``r = x - D(z)`` (so the flow works in a small K-dim coefficient space, not raw
pixels), and ``flow`` is a conditional RealNVP normalizing flow giving exact
likelihood ``p(c | z)``. Averaging many samples recovers ``D(z)`` (the flow is a
proper conditional distribution whose mean is the regression mean); individual
samples restore the detail the mean threw away.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ._train import resolve_device


class MeanDecoder(nn.Module):
    """MLP regressing an embedding coordinate ``z`` -> image ``x`` (E[x|z])."""

    def __init__(self, out_dim: int, n_components: int = 2, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_components, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _CondCoupling(nn.Module):
    """One conditional affine-coupling layer of a RealNVP flow."""

    def __init__(self, dim: int, cond: int, hidden: int = 128, mask_even: bool = True):
        super().__init__()
        mask = (torch.arange(dim) % 2 == 0).float()
        if not mask_even:
            mask = 1.0 - mask
        self.register_buffer("mask", mask)
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim + cond, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * dim),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        xm = x * self.mask
        st = self.net(torch.cat([xm, z], 1))
        s, t = st[:, : self.dim], st[:, self.dim :]
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        y = xm + (1 - self.mask) * (x * torch.exp(s) + t)
        return y, s.sum(1)

    def inverse(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        ym = y * self.mask
        st = self.net(torch.cat([ym, z], 1))
        s, t = st[:, : self.dim], st[:, self.dim :]
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        return ym + (1 - self.mask) * ((y - t) * torch.exp(-s))


class CondFlow(nn.Module):
    """Conditional RealNVP over residual coefficients, conditioned on ``z``."""

    def __init__(self, dim: int, cond: int = 2, layers: int = 10):
        super().__init__()
        self.dim = dim
        self.couplings = nn.ModuleList(
            [_CondCoupling(dim, cond, mask_even=(i % 2 == 0)) for i in range(layers)]
        )

    def nll(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        ld = x.new_zeros(x.shape[0])
        for c in self.couplings:
            x, d = c(x, z)
            ld = ld + d
        return (0.5 * (x**2).sum(1) - ld).mean() + 0.5 * x.shape[1] * np.log(2 * np.pi)

    def log_prob(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        ld = x.new_zeros(x.shape[0])
        for c in self.couplings:
            x, d = c(x, z)
            ld = ld + d
        return -(0.5 * (x**2).sum(1)) + ld - 0.5 * x.shape[1] * np.log(2 * np.pi)

    def sample(self, z: torch.Tensor, n_per: int = 1, temperature: float = 1.0):
        z = z.repeat_interleave(n_per, 0)
        y = torch.randn(z.shape[0], self.dim, device=z.device) * temperature
        for c in reversed(self.couplings):
            y = c.inverse(y, z)
        return y


class GenerativeDecoder:
    """Mean decoder + residual-PCA + conditional flow => samples of ``p(x | z)``.

    Fit on ``(z, x)`` pairs (embeddings of the training data and the raw images).
    Exposes ``mean(z)`` (E[x|z]), ``sample(z, n_per, temperature)`` (draws), and
    ``log_prob(x, z)`` (a manifold-consistency score in residual space).
    """

    def __init__(
        self,
        n_components: int = 2,
        residual_dim: int = 15,
        flow_layers: int = 10,
        hidden: int = 128,
        residual_mode: str = "pca",
        device: str | None = None,
    ):
        if residual_mode not in ("pca", "full"):
            raise ValueError("residual_mode must be 'pca' or 'full'")
        self.n_components = int(n_components)
        self.residual_dim = int(residual_dim)
        self.flow_layers = int(flow_layers)
        self.hidden = int(hidden)
        self.residual_mode = residual_mode
        self.device = resolve_device(device)
        self.dec: MeanDecoder | None = None
        self.flow: CondFlow | None = None

    def fit(
        self,
        Z: np.ndarray,
        X: np.ndarray,
        dec_epochs: int = 500,
        flow_epochs: int = 600,
        batch_size: int = 256,
        lr_dec: float = 3e-3,
        lr_flow: float = 2e-3,
        seed: int = 0,
        verbose: bool = False,
    ) -> "GenerativeDecoder":
        torch.manual_seed(seed)
        np.random.seed(seed)
        Zt = torch.as_tensor(np.asarray(Z, np.float32), device=self.device)
        Xt = torch.as_tensor(np.asarray(X, np.float32), device=self.device)
        out_dim = Xt.shape[1]

        # (1) mean decoder
        self.dec = MeanDecoder(out_dim, self.n_components, self.hidden).to(self.device)
        opt = torch.optim.Adam(self.dec.parameters(), lr=lr_dec)
        for ep in range(dec_epochs):
            opt.zero_grad()
            loss = ((self.dec(Zt) - Xt) ** 2).mean()
            loss.backward()
            opt.step()
        self.dec.eval()

        # (2) residual representation: PCA subspace (default) or full-rank.
        # 'pca'  -> project residuals onto top-K principal directions (compact,
        #           but truncation is a low-rank tell a discriminator can exploit).
        # 'full' -> model all residual dims directly (V = identity); a higher-rank
        #           flow that removes the low-rank tell at the source.
        with torch.no_grad():
            R = (Xt - self.dec(Zt)).cpu().numpy()
        self.res_mean_ = R.mean(0).astype(np.float32)
        Rc = R - self.res_mean_
        if self.residual_mode == "full":
            K = R.shape[1]
            self.V_ = np.eye(K, dtype=np.float32)  # identity: model every dim
            C = Rc.astype(np.float32)
        else:
            _, _, Vt = np.linalg.svd(Rc, full_matrices=False)
            K = min(self.residual_dim, Vt.shape[0])
            self.V_ = Vt[:K].T.astype(np.float32)  # (out_dim, K)
            C = (Rc @ Vt[:K].T).astype(np.float32)
        self.residual_dim = K
        self.c_std_ = (C.std(0) + 1e-3).astype(np.float32)
        Cn = torch.as_tensor(C / self.c_std_, device=self.device)

        # (3) conditional flow on normalized residual coeffs
        self.flow = CondFlow(K, cond=self.n_components, layers=self.flow_layers).to(
            self.device
        )
        optf = torch.optim.Adam(self.flow.parameters(), lr=lr_flow)
        n = len(Cn)
        for ep in range(flow_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                optf.zero_grad()
                loss = self.flow.nll(Cn[idx], Zt[idx])
                loss.backward()
                optf.step()
            if verbose and (ep % 100 == 0 or ep == flow_epochs - 1):
                print(f"flow epoch={ep} nll={float(loss):.3f}")
        self.flow.eval()
        return self

    def mean(self, Z: np.ndarray) -> np.ndarray:
        Zt = torch.as_tensor(np.asarray(Z, np.float32), device=self.device)
        with torch.no_grad():
            return self.dec(Zt).cpu().numpy()

    def sample(
        self, Z: np.ndarray, n_per: int = 1, temperature: float = 1.0, seed: int | None = None
    ) -> np.ndarray:
        """Return (n_query, n_per, out_dim) draws from ``p(x | z)``."""
        if seed is not None:
            torch.manual_seed(seed)
        Zt = torch.as_tensor(np.asarray(Z, np.float32), device=self.device)
        V = torch.as_tensor(self.V_, device=self.device)
        cstd = torch.as_tensor(self.c_std_, device=self.device)
        rmean = torch.as_tensor(self.res_mean_, device=self.device)
        with torch.no_grad():
            cn = self.flow.sample(Zt, n_per=n_per, temperature=temperature)
            r = cn * cstd @ V.T + rmean
            base = self.dec(Zt).repeat_interleave(n_per, 0)
            x = base + r
        return x.cpu().numpy().reshape(len(Zt), n_per, -1)

    def log_prob(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Residual-space log-density log p(c|z): a manifold-consistency score."""
        Zt = torch.as_tensor(np.asarray(Z, np.float32), device=self.device)
        Xt = torch.as_tensor(np.asarray(X, np.float32), device=self.device)
        V = torch.as_tensor(self.V_, device=self.device)
        cstd = torch.as_tensor(self.c_std_, device=self.device)
        rmean = torch.as_tensor(self.res_mean_, device=self.device)
        with torch.no_grad():
            r = Xt - self.dec(Zt) - rmean
            cn = (r @ V) / cstd
            return self.flow.log_prob(cn, Zt).cpu().numpy()

    def state_dict(self) -> dict:
        return {
            "n_components": self.n_components,
            "residual_dim": self.residual_dim,
            "flow_layers": self.flow_layers,
            "hidden": self.hidden,
            "residual_mode": self.residual_mode,
            "out_dim": int(self.V_.shape[0]),
            "dec": {k: v.cpu().numpy() for k, v in self.dec.state_dict().items()},
            "flow": {k: v.cpu().numpy() for k, v in self.flow.state_dict().items()},
            "V": self.V_,
            "c_std": self.c_std_,
            "res_mean": self.res_mean_,
        }

    @classmethod
    def load_state(cls, sd: dict, device: str | None = None) -> "GenerativeDecoder":
        obj = cls(
            n_components=sd["n_components"],
            residual_dim=sd["residual_dim"],
            flow_layers=sd["flow_layers"],
            hidden=sd["hidden"],
            residual_mode=sd.get("residual_mode", "pca"),
            device=device,
        )
        obj.dec = MeanDecoder(sd["out_dim"], sd["n_components"], sd["hidden"]).to(obj.device)
        obj.dec.load_state_dict(
            {k: torch.as_tensor(v) for k, v in sd["dec"].items()}
        )
        obj.dec.eval()
        obj.flow = CondFlow(sd["residual_dim"], cond=sd["n_components"], layers=sd["flow_layers"]).to(
            obj.device
        )
        obj.flow.load_state_dict(
            {k: torch.as_tensor(v) for k, v in sd["flow"].items()}
        )
        obj.flow.eval()
        obj.V_ = np.asarray(sd["V"], np.float32)
        obj.c_std_ = np.asarray(sd["c_std"], np.float32)
        obj.res_mean_ = np.asarray(sd["res_mean"], np.float32)
        return obj
