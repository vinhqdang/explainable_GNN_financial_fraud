"""Sanity check of the Elliptic data pipeline against Weber et al. (2019).

Weber et al. train on time steps 1-34, test on 35-49 and use a 0.5 threshold
(random forest: 50 trees, 50 features per split). This script reproduces that
protocol and, for comparison, trains the same forest on steps 1-30 only (the
training range of the manuscript protocol). Results are printed, not stored.
"""
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtxgnn.data import load_elliptic  # noqa: E402


def report(name, y, p):
    print(f"{name:40s} P={precision_score(y, p):.3f} R={recall_score(y, p):.3f} F1={f1_score(y, p):.3f}")


def main():
    d = load_elliptic()
    x, y, t = d.x.numpy(), d.y.numpy(), d.time_step.numpy()
    lab = y >= 0
    test = lab & (t >= 35)
    for last_train in (34, 30):
        train = lab & (t <= last_train)
        for seed in range(3):
            rf = RandomForestClassifier(n_estimators=50, max_features=50, random_state=seed, n_jobs=4)
            rf.fit(x[train], y[train])
            report(f"RF, train 1-{last_train}, thr 0.5, seed {seed}", y[test], rf.predict(x[test]))
    train = lab & (t <= 34)
    rf = RandomForestClassifier(n_estimators=50, max_features=50, random_state=0, n_jobs=4)
    rf.fit(x[train][:, :93], y[train])
    report("RF local features, train 1-34, thr 0.5", y[test], rf.predict(x[test][:, :93]))
    lr = LogisticRegression(max_iter=2000).fit(x[train], y[train])
    report("LR, train 1-34, thr 0.5", y[test], lr.predict(x[test]))


if __name__ == "__main__":
    main()
