# src/models/esn.py
import numpy as np

class EchoStateNetwork:
    """
    Echo State Network (ESN) with leaky-integrator reservoir and ridge readout.

    API:
        esn = EchoStateNetwork(hidden_size=500, spectral_radius=0.9, leak_rate=1.0, ridge_alpha=1.0)
        esn.fit(X_train, y_train)
        y_hat = esn.predict(X_test)

    Notes
    -----
    - Inputs X are expected to be already standardized (your folds produce z_* features).
    - Targets y can be shape (n_samples,) or (n_samples, n_outputs).
    - We collect reservoir states after a warmup "washout" and fit a ridge readout:
          H = [1, x_t]^T    (bias + reservoir state)
          W_out = argmin ||H W_out - Y||^2 + alpha * ||W_out||^2
    - By default, prediction starts from the last training state (continue_state=True),
      which is appropriate for contiguous walk-forward splits (no leakage).
    """

    def __init__(
        self,
        hidden_size: int = 500,
        spectral_radius: float = 0.9,
        leak_rate: float = 1.0,
        input_scale: float = 1.0,
        bias_scale: float = 0.2,
        density: float = 0.1,
        ridge_alpha: float = 1.0,
        washout: int = 100,
        continue_state: bool = True,
        state_clip: float | None = None,
        seed: int = 0,
        power_iter: int = 100,
    ):
        """
        Parameters
        ----------
        hidden_size : number of reservoir units.
        spectral_radius : target spectral radius of recurrent matrix W (<= 1.0 typical).
        leak_rate : leaky integrator rate 'a' (0<a<=1). Lower = slower reservoir.
        input_scale : scale for input weights W_in.
        bias_scale : scale for bias column in W_in.
        density : fraction of non-zero entries in W (sparse random reservoir).
        ridge_alpha : L2 regularization strength for readout.
        washout : number of initial time steps to discard when collecting states.
        continue_state : if True, start predict() from last train state; else zeros.
        state_clip : if set, clip reservoir states to [-state_clip, state_clip] each step.
        seed : RNG seed for reproducibility.
        power_iter : iterations used to estimate spectral radius via power iteration.
        """
        self.hidden_size   = int(hidden_size)
        self.spectral_radius = float(spectral_radius)
        self.leak_rate     = float(leak_rate)
        self.input_scale   = float(input_scale)
        self.bias_scale    = float(bias_scale)
        self.density       = float(density)
        self.ridge_alpha   = float(ridge_alpha)
        self.washout       = int(max(0, washout))
        self.continue_state = bool(continue_state)
        self.state_clip    = state_clip
        self.seed          = int(seed)
        self.power_iter    = int(power_iter)

        # set at fit-time
        self._rng = np.random.default_rng(self.seed)
        self.W_in = None      # (H, D+1) including bias
        self.W    = None      # (H, H)
        self.W_out = None     # (H+1, O)
        self.last_state_ = None
        self.input_dim_ = None
        self.output_dim_ = None
        self._fitted = False

    # --------- public API ---------
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the readout on reservoir states generated from X."""
        X = self._check_X(X)
        Y = self._check_y(y)
        n, d = X.shape
        _, o = Y.shape

        if self.W is None:
            self._init_weights(input_dim=d)

        # roll reservoir over the entire train sequence
        states = self._collect_states(X)  # shape (n, H)

        # discard initial washout steps
        if self.washout >= states.shape[0]:
            raise ValueError(f"washout={self.washout} >= n_samples={states.shape[0]}. Reduce washout.")
        H = states[self.washout:, :]             # (n_eff, H)
        Y_eff = Y[self.washout:, :]              # (n_eff, O)

        # add bias column to state design matrix
        H_aug = np.concatenate([np.ones((H.shape[0], 1)), H], axis=1)  # (n_eff, H+1)

        # ridge closed-form: (H^T H + alpha I)^{-1} H^T Y
        # regularize all weights incl. bias for simplicity
        A = H_aug.T @ H_aug
        A += self.ridge_alpha * np.eye(A.shape[0])
        B = H_aug.T @ Y_eff
        self.W_out = np.linalg.solve(A, B)       # (H+1, O)

        # persist last state (to start prediction if continue_state=True)
        self.last_state_ = states[-1].copy()
        self.input_dim_  = d
        self.output_dim_ = o
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for a new sequence X."""
        self._require_fitted()
        X = self._check_X(X)
        if X.shape[1] != self.input_dim_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.input_dim_}.")

        # choose initial state
        x = np.zeros(self.hidden_size) if not self.continue_state or self.last_state_ is None \
            else self.last_state_.copy()

        # roll forward and compute outputs
        Y_hat = np.zeros((X.shape[0], self.output_dim_))
        for t in range(X.shape[0]):
            x = self._step(x, X[t])
            if self.state_clip is not None:
                np.clip(x, -self.state_clip, self.state_clip, out=x)
            # output = [1, x]^T W_out
            y_t = np.concatenate(([1.0], x)) @ self.W_out
            Y_hat[t, :] = y_t

        # do NOT update last_state_ here (keeps fit/predict behavior pure)
        return Y_hat.ravel() if self.output_dim_ == 1 else Y_hat

    # --------- core mechanics ---------
    def _init_weights(self, input_dim: int):
        H = self.hidden_size
        D = input_dim
        rng = self._rng

        # Input weights (H, D+1): first column is bias input = 1
        self.W_in = np.empty((H, D + 1), dtype=float)
        self.W_in[:, 0] = rng.uniform(-self.bias_scale, self.bias_scale, size=H)
        self.W_in[:, 1:] = rng.uniform(-self.input_scale, self.input_scale, size=(H, D))

        # Recurrent reservoir W (sparse, scaled to spectral_radius)
        W = np.zeros((H, H), dtype=float)
        nnz = int(self.density * H * H)
        if nnz > 0:
            rows = rng.integers(0, H, size=nnz)
            cols = rng.integers(0, H, size=nnz)
            vals = rng.uniform(-1.0, 1.0, size=nnz)
            W[rows, cols] = vals

        # Scale to target spectral radius using power iteration (stable for larger H)
        radius = self._power_iteration_radius(W, iters=self.power_iter)
        if radius > 0:
            W *= (self.spectral_radius / radius)
        self.W = W

    def _collect_states(self, X: np.ndarray) -> np.ndarray:
        """Run reservoir over X and return state matrix of shape (n, H)."""
        n, _ = X.shape
        H = self.hidden_size
        states = np.zeros((n, H), dtype=float)
        x = np.zeros(H, dtype=float)
        for t in range(n):
            x = self._step(x, X[t])
            if self.state_clip is not None:
                np.clip(x, -self.state_clip, self.state_clip, out=x)
            states[t, :] = x
        return states

    def _step(self, x_prev: np.ndarray, u_t: np.ndarray) -> np.ndarray:
        """
        Leaky update:
            x_t = (1 - a) * x_{t-1} + a * tanh( W_in [1; u_t] + W x_{t-1} )
        """
        a = self.leak_rate
        # net input: bias + inputs
        in_vec = np.concatenate(([1.0], u_t))            # (D+1,)
        preact = self.W_in @ in_vec + self.W @ x_prev    # (H,)
        x = (1.0 - a) * x_prev + a * np.tanh(preact)
        return x

    # --------- utils ---------
    def _power_iteration_radius(self, W: np.ndarray, iters: int = 100) -> float:
        """Estimate spectral radius via power iteration (fast, memory-friendly)."""
        if W.size == 0:
            return 0.0
        v = self._rng.uniform(-1.0, 1.0, size=(W.shape[0],))
        v /= (np.linalg.norm(v) + 1e-12)
        for _ in range(max(1, iters)):
            v = W @ v
            nrm = np.linalg.norm(v)
            if nrm == 0:
                return 0.0
            v /= nrm
        # Rayleigh quotient approximation of dominant eigenvalue magnitude
        wv = W @ v
        num = float(v @ wv)
        den = float(v @ v) + 1e-12
        radius = abs(num / den)
        return radius

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        return X

    def _check_y(self, y: np.ndarray) -> np.ndarray:
        Y = np.asarray(y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.ndim != 2:
            raise ValueError(f"y must be 1D or 2D, got shape {y.shape}.")
        return Y

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("ESN is not fitted. Call .fit(X, y) first.")

    # --------- convenience accessors ---------
    @property
    def coef_(self):
        """Return readout weights excluding bias (shape (H, O))."""
        if self.W_out is None:
            return None
        return self.W_out[1:, :]

    @property
    def intercept_(self):
        """Return readout bias term(s) (shape (O,))."""
        if self.W_out is None:
            return None
        b = self.W_out[0, :]
        return b.ravel()
