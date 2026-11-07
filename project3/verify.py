"""Recompute published metrics from the frozen predictions and replay selected test images."""
from pathlib import Path
import csv
import json

import numpy as np
import torch

from train import ROOT, GROUPS, make_model, tensor_images


def main():
    out = ROOT/'results'
    design = json.loads((ROOT/'protocol.json').read_text())
    seeds = [design['primary_training_seed'], *design['replication_training_seeds']]
    with np.load(ROOT/'data/cifar10.npz') as ds:
        test_y, train_y = ds['y_test'], ds['y_train']
        x = ds['x_test']
    assert np.array_equal(x,np.load(out/'test_images.npy',mmap_mode='r'))
    rng = np.random.default_rng(18)
    probe = np.sort(rng.choice(len(x),64,replace=False))
    verified = []
    torch.set_num_threads(2)
    for seed in seeds:
        d = out if seed == seeds[0] else out/'replicates'/str(seed)
        with np.load(d/'split.npz') as split, np.load(out/'split.npz') as ref:
            assert all(np.array_equal(split[k],ref[k]) for k in ('train','val'))
            assert len(np.intersect1d(split['train'],split['val'])) == 0
            assert np.all(np.bincount(train_y[split['train']])==4500) and np.all(np.bincount(train_y[split['val']])==500)
        for group,(_,attention) in GROUPS.items():
            gd = d/group
            cfg = json.loads((gd/'config.json').read_text())
            test = json.loads((gd/'test.json').read_text())
            with (gd/'history.csv').open() as f: history=list(csv.DictReader(f))
            assert len(history)==design['epochs_per_run'] and cfg['seed']==seed
            epoch = 1+np.argmax([float(r['val_acc']) for r in history])
            assert epoch==test['best_epoch']
            with np.load(gd/'predictions.npz') as pred:
                y, yh, conf = pred['y_true'], pred['y_pred'], pred['confidence']
            assert np.array_equal(y,test_y)
            assert np.all((yh>=0)&(yh<10)) and np.all((conf>=.1-1e-6)&(conf<=1))
            matrix = np.bincount(y*10+yh,minlength=100).reshape(10,10)
            assert np.all(matrix.sum(1)==1000)
            assert abs(matrix.trace()/10000 - test['test_acc']) < 1e-6
            weights = torch.load(gd/'best.pt',map_location='cpu',weights_only=True)
            assert weights['epoch']==epoch and abs(weights['val_acc']-test['val_acc'])<1e-7
            model = make_model(attention).eval()
            model.load_state_dict(weights['model'])
            with torch.no_grad():
                prob = model(tensor_images(x[probe],'cpu')).softmax(1)
            np.testing.assert_array_equal(prob.argmax(1).numpy(),yh[probe])
            np.testing.assert_allclose(prob.max(1).values.numpy(),conf[probe],atol=.0002,rtol=.0002)
            verified.append(dict(seed=seed,group=group,test_count=len(y),correct=int(matrix.trace()),
                                 best_epoch=int(epoch),cpu_replay_count=len(probe)))
    (out/'verification.json').write_text(json.dumps(verified,indent=2)+'\n')
    print(f'Verified {len(verified)} runs, 10000 predictions/run and {len(probe)} CPU checkpoint replays/run.')


if __name__=='__main__':
    main()
