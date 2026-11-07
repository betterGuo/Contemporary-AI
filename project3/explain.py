"""Matched Grad-CAM panels, using the same image and true-class logit for all groups."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from train import GROUPS, ROOT, make_model, tensor_images


def main():
    torch.set_num_threads(2)
    out = ROOT/'results'
    dest = out/'explanations'
    dest.mkdir(exist_ok=True)
    names = json.loads((ROOT/'data/source.json').read_text())['class_names']
    images = np.load(out/'test_images.npy', mmap_mode='r')
    preds = {}
    for g in GROUPS:
        with np.load(out/g/'predictions.npz') as f:
            y = f['y_true']
            preds[g] = f['y_pred']
    correct = np.stack([preds[g] == y for g in GROUPS])
    rules = {'combined fixes baseline': ~correct[0] & correct[-1],
             'combined loses baseline': correct[0] & ~correct[-1],
             'all four incorrect': ~correct.any(axis=0),
             'all four correct': correct.all(axis=0)}
    selected = [(rule,int(np.flatnonzero(mask)[0])) for rule, mask in rules.items() if mask.any()]
    fig, axes = plt.subplots(len(selected), 5, figsize=(12, 2.8*len(selected)), squeeze=False,
                             layout='constrained')
    fig.suptitle('Matched Grad-CAM | target = true class | common 8 x 8 feature layer', fontsize=14)
    records, maps = [], {}
    for i,(rule,idx) in enumerate(selected):
        axes[i,0].imshow(images[idx], interpolation='nearest')
        axes[i,0].set_title(f'{rule}\n#{idx} | true: {names[y[idx]]}', fontsize=9)
    for col,(group,(_, att)) in enumerate(GROUPS.items(),1):
        m = make_model(attention=att)
        m.load_state_dict(torch.load(out/group/'best.pt', map_location='cpu',weights_only=True)['model'])
        m.eval()
        captured = {}
        def hook(_module, _inputs, features):
            captured['features'] = features
        handle = m.gate.register_forward_hook(hook)
        for i, (rule,idx) in enumerate(selected):
            sample = np.array(images[idx:idx+1])
            logits = m(tensor_images(sample, 'cpu'))
            activation = captured['features']
            grad, = torch.autograd.grad(logits[0,int(y[idx])], activation)
            cam = (activation*grad.mean((2,3), keepdim=True)).sum(1, keepdim=True).clamp_min(0)
            cam = F.interpolate(cam, size=(32,32), mode='bilinear',align_corners=False)[0,0].detach().numpy()
            cam /= max(float(cam.max()), 1e-12)
            maps[f'{group}_{idx}'] = cam
            pred = int(preds[group][idx])
            ax = axes[i,col]
            ax.imshow(sample[0],interpolation='nearest')
            ax.imshow(cam, cmap='inferno', alpha=.52, vmin=0, vmax=1)
            ax.set_title(f'{group}\npred: {names[pred]}',fontsize=9)
            records.append(dict(rule=rule,test_index=idx,group=group,true_class=names[y[idx]],
                                predicted_class=names[pred],target_logit=float(logits[0,int(y[idx])].detach())))
        handle.remove()
    for ax in axes.flat: ax.axis('off')
    fig.savefig(dest/'matched_gradcam.png',dpi=180)
    plt.close(fig)
    np.savez_compressed(dest/'heatmaps.npz', **maps)
    with (dest/'samples.csv').open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
    print('Saved', len(records), 'matched maps from',len(selected),'test images.')


if __name__=='__main__':
    main()
