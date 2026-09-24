import torch, numpy as np, sys
torch.set_num_threads(2)
from common import elliptic
from rtxgnn.graph import split_regions
from rtxgnn.train import build, train_model, evaluate, set_seed
from sklearn.preprocessing import QuantileTransformer
d = elliptic()
tr = (d.t <= 30).numpy()
for mode in sys.argv[1].split(','):
    x = d.x.numpy().copy()
    if mode == 'qt':
        q = QuantileTransformer(output_distribution='normal', n_quantiles=1000, subsample=200000, random_state=0).fit(x[tr])
        x = q.transform(x)
    d.x = torch.tensor(x, dtype=torch.float)
    dtr, dev = split_regions(d)
    for name in sys.argv[2].split(','):
        for seed in [0,1]:
            set_seed(seed); m = build(name, 165)
            p, thr, h = train_model(m, dtr, dev)
            print(mode, name, seed, {k: round(v,4) for k,v in evaluate(p, dev, thr).items()}, flush=True)
