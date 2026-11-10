import json

import numpy as np
import pandas as pd
import xgboost as xgb


def load_model():
    booster = xgb.Booster()
    booster.load_model("model/xgb_booster.json")
    with open("model/encoding.json") as f:
        enc = json.load(f)
    return booster, enc


def encode(X, enc):
    """Turn raw inputs into the 49 model columns, in the same order as training."""
    cols, names = [], []
    for c in enc["cat_cols"]:
        vals = X[c].astype(str)
        for cat in enc["categories"][c]:
            cols.append((vals == cat).astype(float).values)
            names.append(c)
    for c in enc["num_cols"]:
        cols.append(X[c].astype(float).values)
        names.append(c)
    return np.column_stack(cols), names


def predict(booster, enc, X):
    """Risk score + SHAP per driver (one-hot pieces added back together)."""
    M, names = encode(X, enc)
    d = xgb.DMatrix(M)
    p = booster.predict(d)
    contrib = booster.predict(d, pred_contribs=True)[:, :-1]    # last column = starting value
    shap = pd.DataFrame(contrib, columns=names).T.groupby(level=0).sum().T
    return p, shap