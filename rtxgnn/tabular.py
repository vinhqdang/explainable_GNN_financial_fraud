"""Feature-only baselines (no graph): logistic regression, random forest, XGBoost."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def fit_tabular(name, X_tr, y_tr, seed=0):
    if name == "lr":
        m = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    elif name == "rf":
        # configuration of Weber et al. (2019)
        m = RandomForestClassifier(n_estimators=100, max_features=50, n_jobs=-1, random_state=seed)
    elif name == "xgb":
        from xgboost import XGBClassifier
        pos = max(1, int(y_tr.sum()))
        m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                          colsample_bytree=0.8, scale_pos_weight=(len(y_tr) - pos) / pos,
                          n_jobs=4, random_state=seed, verbosity=0)
    m.fit(X_tr, y_tr)
    return m


def predict(m, X):
    return m.predict_proba(X)[:, 1]
