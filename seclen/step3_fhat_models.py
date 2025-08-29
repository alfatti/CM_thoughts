# step3_fhat_models.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Sequence, Tuple, Optional, Dict
import numpy as np

# PyTorch MLP
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Sklearn boosted tree (no external deps)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib

from step1_clusters import ActionCatalog, Cluster
from step3_datasets import FhatDesignMatrixBuilder

# ---------------------
# Torch MLP definition
# ---------------------
class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, hidden2: Optional[int] = 128, dropout: float = 0.1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        if hidden2:
            layers += [nn.Linear(hidden, hidden2), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers += [nn.Linear(hidden2, 1)]
        else:
            layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B,1]

# ---------------------
# Unified Fhat wrapper
# ---------------------

@dataclass
class FhatConfig:
    kind: Literal["mlp", "hgb"] = "mlp"
    # MLP
    hidden: int = 256
    hidden2: int = 128
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 512
    epochs: int = 20
    huber_delta: float = 1.0
    weight_decay: float = 0.0
    # HGB (HistGradientBoostingRegressor)
    hgb_max_depth: Optional[int] = None
    hgb_learning_rate: float = 0.05
    hgb_max_iter: int = 400
    hgb_early_stopping: bool = True
    hgb_validation_fraction: float = 0.1
    hgb_l2_regularization: float = 0.0

class FhatModel:
    """
    A unified interface over two backends:
      • 'mlp'  -> PyTorch MLP
      • 'hgb'  -> sklearn HistGradientBoostingRegressor
    Exposes: fit(X, A, y), predict(X, A), save(path), load(path), and
             cluster_argmax/value via a scorer helper.
    """
    def __init__(self, config: FhatConfig):
        self.cfg = config
        self.kind = config.kind
        self._mlp: Optional[_MLP] = None
        self._hgb: Optional[HistGradientBoostingRegressor] = None
        self._in_dim: Optional[int] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, A: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, A_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        X: [N, d_x], A: [N, d_a], y: [N,1]
        """
        Z = np.hstack([X, A]).astype(np.float32)
        self._in_dim = Z.shape[1]

        if self.kind == "mlp":
            return self._fit_mlp(Z, y, Z_val=None if X_val is None else np.hstack([X_val, A_val]), y_val=y_val)
        else:
            return self._fit_hgb(Z, y, Z_val=None if X_val is None else np.hstack([X_val, A_val]), y_val=y_val)

    def predict(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        Z = np.hstack([X, A]).astype(np.float32)
        if self.kind == "mlp":
            self._ensure_mlp()
            with torch.no_grad():
                tZ = torch.from_numpy(Z).to(self.device)
                preds = self._mlp(tZ).cpu().numpy()
            return preds  # [N,1]
        else:
            self._ensure_hgb()
            return self._hgb.predict(Z).reshape(-1, 1)

    # -------- cluster utilities (scoring K candidate actions for one or many X) --------

    def cluster_argmax(self,
                       dm: FhatDesignMatrixBuilder,
                       X_batch: np.ndarray,
                       catalog: ActionCatalog,
                       cluster: Cluster) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each row in X_batch, return:
          • best_action_id in the cluster
          • best_value = fhat(x, best_action)
        Shapes: X_batch [B, d_x], returns (best_ids [B], best_vals [B,1])
        """
        B = X_batch.shape[0]
        best_ids = np.empty(B, dtype=int)
        best_vals = np.empty((B, 1), dtype=np.float32)

        cand_aids = catalog.actions_in_cluster(cluster)
        for i in range(B):
            X_row = X_batch[i]
            X_tile, A_tile = dm.tile_and_stack(X_row, cand_aids)  # [K,d_x], [K,d_a]
            scores = self.predict(X_tile, A_tile)                 # [K,1]
            k = int(np.argmax(scores))
            best_ids[i] = cand_aids[k]
            best_vals[i, 0] = scores[k, 0]
        return best_ids, best_vals

    def cluster_value(self,
                      dm: FhatDesignMatrixBuilder,
                      X_batch: np.ndarray,
                      catalog: ActionCatalog,
                      cluster: Cluster) -> np.ndarray:
        """
        Return [B,1] values = max_{a in cluster} fhat(x,a)
        """
        _, vals = self.cluster_argmax(dm, X_batch, catalog, cluster)
        return vals

    # -------- persistence --------

    def save(self, path: str):
        meta = {"cfg": self.cfg.__dict__, "kind": self.kind, "in_dim": self._in_dim}
        if self.kind == "mlp":
            self._ensure_mlp()
            torch.save({"meta": meta, "state_dict": self._mlp.state_dict()}, path)
        else:
            self._ensure_hgb()
            joblib.dump({"meta": meta, "sk_model": self._hgb}, path)

    @staticmethod
    def load(path: str) -> "FhatModel":
        # try torch first
        try:
            blob = torch.load(path, map_location="cpu")
            meta = blob["meta"]
            cfg = FhatConfig(**meta["cfg"])
            m = FhatModel(cfg)
            if meta["kind"] != "mlp":
                raise RuntimeError("File is not an MLP checkpoint")
            net = _MLP(meta["in_dim"], cfg.hidden, cfg.hidden2, cfg.dropout)
            net.load_state_dict(blob["state_dict"])
            m._mlp = net.eval()
            m._in_dim = meta["in_dim"]
            return m
        except Exception:
            # try joblib sklearn
            obj = joblib.load(path)
            meta = obj["meta"]
            cfg = FhatConfig(**meta["cfg"])
            m = FhatModel(cfg)
            if meta["kind"] != "hgb":
                raise RuntimeError("File is not an HGB checkpoint")
            m._hgb = obj["sk_model"]
            m._in_dim = meta["in_dim"]
            return m

    # -------- internals --------

    def _fit_mlp(self, Z: np.ndarray, y: np.ndarray,
                 Z_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> Dict[str, float]:
        self._mlp = _MLP(self._in_dim, self.cfg.hidden, self.cfg.hidden2, self.cfg.dropout).to(self.device)
        opt = torch.optim.Adam(self._mlp.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        loss_fn = nn.HuberLoss(delta=self.cfg.huber_delta)

        ds = TensorDataset(torch.from_numpy(Z), torch.from_numpy(y))
        dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False)

        best_val = float("inf")
        history = {}
        for epoch in range(self.cfg.epochs):
            self._mlp.train()
            epoch_loss = 0.0
            for (bZ, by) in dl:
                bZ = bZ.to(self.device).float()
                by = by.to(self.device).float()
                pred = self._mlp(bZ)
                loss = loss_fn(pred, by)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._mlp.parameters(), 5.0)
                opt.step()
                epoch_loss += loss.item() * bZ.size(0)
            train_rmse = float(np.sqrt(epoch_loss / len(ds)))

            if Z_val is not None:
                self._mlp.eval()
                with torch.no_grad():
                    pv = self._mlp(torch.from_numpy(Z_val).to(self.device)).cpu().numpy()
                val_rmse = float(np.sqrt(np.mean((pv - y_val) ** 2)))
            else:
                val_rmse = train_rmse

            history[f"epoch_{epoch}"] = {"train_rmse": train_rmse, "val_rmse": val_rmse}
            if val_rmse < best_val:
                best_val = val_rmse
                best_state = {k: v.cpu() for k, v in self._mlp.state_dict().items()}
            # simple early keep-best
        self._mlp.load_state_dict(best_state)  # type: ignore
        self._mlp.eval()
        return {"val_rmse": best_val}

    def _fit_hgb(self, Z: np.ndarray, y: np.ndarray,
                 Z_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> Dict[str, float]:
        self._hgb = HistGradientBoostingRegressor(
            max_depth=self.cfg.hgb_max_depth,
            learning_rate=self.cfg.hgb_learning_rate,
            max_iter=self.cfg.hgb_max_iter,
            early_stopping=self.cfg.hgb_early_stopping,
            validation_fraction=self.cfg.hgb_validation_fraction,
            l2_regularization=self.cfg.hgb_l2_regularization,
            loss="squared_error",
        )
        self._hgb.fit(Z, y.ravel())
        preds = self._hgb.predict(Z if Z_val is None else Z_val)
        rmse = float(np.sqrt(mean_squared_error(y if y_val is None else y_val, preds.reshape(-1,1))))
        return {"val_rmse": rmse}

    def _ensure_mlp(self):
        if self._mlp is None:
            raise RuntimeError("MLP model not initialized. Did you fit or load?")

    def _ensure_hgb(self):
        if self._hgb is None:
            raise RuntimeError("HGB model not initialized. Did you fit or load?")
