
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- RMSE compatibility across sklearn versions ---
# sklearn >= 1.4 provides root_mean_squared_error; older versions require sqrt(MSE)
try:
    from sklearn.metrics import root_mean_squared_error as _rmse
    def _compute_rmse(y_true, y_pred):
        return float(_rmse(y_true, y_pred))
except Exception:
    def _compute_rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def basic_metrics(y_true, y_pred):
    rmse = _compute_rmse(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    return dict(rmse=rmse, mae=mae, r2=r2, dir_acc=dir_acc)

def evaluate_predictions(y_true, y_pred) -> pd.Series:
    return pd.Series(basic_metrics(y_true, y_pred))

def sign_backtest(y_true, y_pred, cost_per_trade=0.0001):
    pos = np.sign(y_pred)
    pos_change = np.abs(np.diff(pos, prepend=0))
    pnl = pos * y_true - cost_per_trade * pos_change
    ann = np.sqrt(252.0)
    sharpe = pnl.mean() / (pnl.std() + 1e-12) * ann
    hit_ratio = float(np.mean((pos * y_true) > 0))
    return dict(avg_daily_pnl=float(pnl.mean()),
                vol=float(pnl.std()),
                sharpe=float(np.round(sharpe, 3)),
                hit_ratio=float(np.round(hit_ratio, 3)),
                turnover=float(np.round(pos_change.mean(), 3)))
