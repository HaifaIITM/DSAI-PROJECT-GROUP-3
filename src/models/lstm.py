import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class _SeqMaker:
    """Build fixed-length sequences with left-padding so we predict every time step."""
    def __init__(self, seq_len: int):
        self.seq_len = int(seq_len)

    def build(self, X: np.ndarray, y: np.ndarray | None = None):
        # X: (N, F), y: (N,) or (N,O) or None
        N, F = X.shape
        L = self.seq_len
        X_pad = np.vstack([np.repeat(X[0:1], L-1, axis=0), X])  # left-pad with first row
        X_seq = np.lib.stride_tricks.sliding_window_view(X_pad, (L, F))[:, 0, :]
        # X_seq: (N, L, F)
        if y is None:
            return X_seq, None
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, None]
        return X_seq, y

class _LSTMHead(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=1, dropout=0.0, out_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=layers,
                            dropout=(dropout if layers > 1 else 0.0),
                            batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):  # x: (B, L, F)
        _, (hn, _) = self.lstm(x)  # hn: (layers, B, H) → take last layer
        h_last = hn[-1]            # (B, H)
        return self.head(h_last)   # (B, O)

class LSTMRegressor:
    """
    LSTM baseline with left-padded sliding windows.
    .fit(X, y) and .predict(X) accept 2D arrays (N,F) and return aligned predictions (N,) or (N,O).
    """
    def __init__(self,
                 seq_len: int = 32,
                 hidden: int = 128,
                 layers: int = 1,
                 dropout: float = 0.0,
                 epochs: int = 10,
                 batch_size: int = 128,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 val_frac: float = 0.1,
                 seed: int = 0,
                 device: str | None = None):
        self.seq_len = seq_len
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_frac = val_frac
        self.seed = seed
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._seq = _SeqMaker(seq_len)
        self.out_dim_ = None
        self.in_dim_ = None
        self._fitted = False
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        self.out_dim_ = y.shape[1]
        self.in_dim_ = X.shape[1]

        X_seq, y_seq = self._seq.build(X, y)  # (N, L, F), (N, O)

        # simple chronological split: last val_frac as validation
        N = X_seq.shape[0]
        n_val = max(1, int(self.val_frac * N))
        n_tr  = N - n_val
        X_tr, Y_tr = X_seq[:n_tr], y_seq[:n_tr]
        X_va, Y_va = X_seq[n_tr:], y_seq[n_tr:]

        model = _LSTMHead(in_dim=self.in_dim_, hidden=self.hidden, layers=self.layers,
                          dropout=self.dropout, out_dim=self.out_dim_).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        def make_loader(Xa, Ya, bs, shuffle):
            ds = TensorDataset(torch.from_numpy(Xa), torch.from_numpy(Ya))
            return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)

        tr_loader = make_loader(X_tr, Y_tr, self.batch_size, True)
        va_loader = make_loader(X_va, Y_va, self.batch_size, False)

        best_va = np.inf
        best_state = None
        for ep in range(self.epochs):
            model.train()
            tr_loss = 0.0
            for xb, yb in tr_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                tr_loss += loss.item() * len(xb)
            tr_loss /= len(tr_loader.dataset)

            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb = xb.to(self.device); yb = yb.to(self.device)
                    pred = model(xb)
                    va_loss += loss_fn(pred, yb).item() * len(xb)
            va_loss /= len(va_loader.dataset)

            if va_loss < best_va:
                best_va = va_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        self._model = model
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X = np.asarray(X, dtype=np.float32)
        X_seq, _ = self._seq.build(X, None)  # (N, L, F)
        with torch.no_grad():
            y_hat = self._model(torch.from_numpy(X_seq).to(self.device)).cpu().numpy()
        return y_hat.ravel() if self.out_dim_ == 1 else y_hat

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("LSTMRegressor is not fitted.")
