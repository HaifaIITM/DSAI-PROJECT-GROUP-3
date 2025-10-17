import os, json
import pandas as pd
from sklearn.preprocessing import StandardScaler

def build_splits(index: pd.DatetimeIndex, train_days: int, test_days: int, step_days: int):
    idx = index.sort_values()
    folds = []
    start_ix = 0
    while True:
        tr_start = start_ix
        tr_end   = tr_start + train_days
        te_end   = tr_end + test_days
        if te_end > len(idx):
            break
        folds.append({
            "train": {"start": idx[tr_start].strftime("%Y-%m-%d"),
                      "end":   idx[tr_end - 1].strftime("%Y-%m-%d")},
            "test":  {"start": idx[tr_end].strftime("%Y-%m-%d"),
                      "end":   idx[te_end - 1].strftime("%Y-%m-%d")},
            "details": {"train_days": train_days, "test_days": test_days}
        })
        start_ix += step_days
    return folds

def materialize_fold(df: pd.DataFrame, features, targets, fold, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    tr_s, tr_e = fold["train"]["start"], fold["train"]["end"]
    te_s, te_e = fold["test"]["start"],  fold["test"]["end"]

    tr = df.loc[tr_s:tr_e].dropna(subset=targets).copy()
    te = df.loc[te_s:te_e].dropna(subset=targets).copy()

    scaler = StandardScaler()
    tr_X = scaler.fit_transform(tr[features].values)
    te_X = scaler.transform(te[features].values)

    tr_out = pd.DataFrame(tr_X, index=tr.index, columns=[f"z_{c}" for c in features])
    te_out = pd.DataFrame(te_X, index=te.index, columns=[f"z_{c}" for c in features])

    for c in targets:
        tr_out[c] = tr[c].values
        te_out[c] = te[c].values
    tr_out["Symbol"] = tr["Symbol"].values
    te_out["Symbol"] = te["Symbol"].values

    tr_out.to_csv(os.path.join(out_dir, "train.csv"))
    te_out.to_csv(os.path.join(out_dir, "test.csv"))

    scaler_meta = {
        "features": list(features),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "train_start": tr_s, "train_end": tr_e,
        "test_start": te_s,  "test_end": te_e
    }
    with open(os.path.join(out_dir, "scaler.json"), "w") as f:
        json.dump(scaler_meta, f, indent=2)
