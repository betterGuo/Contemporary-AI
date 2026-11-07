"""Reproduce the validation-only horizontal-flip diagnostic: python3 report/support/flip_consistency.py."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

SUPPORT = Path(__file__).resolve().parent
ROOT = SUPPORT.parents[1]
sys.path.insert(0, str(ROOT))
from model import make_model
from train import GROUPS, tensor_images


@torch.inference_mode()
def paired_counts(model, images, labels):
    same = correct_clean = correct_flipped = 0
    for first in range(0, len(labels), 128):
        x = tensor_images(images[first:first+128], 'cpu')
        # Only the width is reversed. The original and flipped views have the same label.
        predicted = model(torch.cat((x, torch.flip(x, dims=(-1,))), dim=0)).argmax(1)
        clean, flipped = predicted.chunk(2)
        target = torch.as_tensor(labels[first:first+128])
        same += int((clean == flipped).sum())
        correct_clean += int((clean == target).sum())
        correct_flipped += int((flipped == target).sum())
    return same, correct_clean, correct_flipped


def main():
    torch.set_num_threads(2)
    protocol = json.loads((ROOT/'protocol.json').read_text())
    primary = protocol['primary_training_seed']
    results = ROOT/'results'
    with np.load(results/'split.npz') as split:
        val_idx = split['val']
        assert len(np.unique(val_idx)) == len(val_idx) == 5000
        assert not len(np.intersect1d(split['train'], val_idx))
    # Do not load or evaluate official test images or labels.
    with np.load(ROOT/'data/cifar10.npz') as data:
        images, labels = data['x_train'][val_idx], data['y_train'][val_idx]
    assert np.all(np.bincount(labels, minlength=10) == 500)
    n = len(labels)
    rows = []
    for seed in [primary, *protocol['replication_training_seeds']]:
        r = results if seed == primary else results/'replicates'/str(seed)
        with np.load(r/'split.npz') as split:
            np.testing.assert_array_equal(val_idx, split['val'])
        for group, (_, use_attention) in GROUPS.items():
            model = make_model(attention=use_attention).eval()
            weights = torch.load(r/group/'best.pt',map_location='cpu', weights_only=True)
            model.load_state_dict(weights['model'])
            same, clean, flip = paired_counts(model, images, labels)
            assert abs(clean / n - weights['val_acc']) < 0.0001
            rows.append(dict(seed=seed, group=group, checkpoint_epoch=weights['epoch'],
                             n_validation=n, predictions_unchanged=same,
                             correct_clean=clean, correct_flipped=flip,
                             flip_consistency_pct=100*same/n,
                             clean_accuracy_pct=100*clean/n,
                             flipped_accuracy_pct=100*flip/n))
    dst = SUPPORT/'flip_consistency.csv'
    with dst.open('w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for group in GROUPS:
        subset = [row['flip_consistency_pct'] for row in rows if row['group']==group]
        print(group, subset, f'mean={np.mean(subset):.2f}%')
    print(dst)


if __name__ == '__main__':
    main()
