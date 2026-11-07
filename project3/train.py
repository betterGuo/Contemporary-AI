"""CIFAR-10: matched 2x2 factorial experiment (augmentation x channel attention)."""
import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from model import make_model

ROOT = Path(__file__).resolve().parent
GROUPS = {'baseline': (False, False), 'augment': (True, False),
          'attention': (False, True), 'both': (True, True)}
SEED = 20261107


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def checkpoint(path, obj):
    tmp = path.with_suffix('.part')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def split_indices(labels):
    rng = np.random.default_rng(SEED)
    trains, vals = [], []
    for c in range(10):
        idx = rng.permutation(np.flatnonzero(labels == c))
        assert len(idx) == 5000
        vals.append(idx[:500])
        trains.append(idx[500:])
    return np.concatenate(trains), np.sort(np.concatenate(vals))


def tensor_images(arr, device):
    # Explicit contiguous NCHW avoids a channels-last backward stride issue on MPS.
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device).permute(0, 3, 1, 2).contiguous().float().div_(127.5).sub_(1.0)


def augment_batch(padded, idx, rng):
    # Each image gets its own 32x32 crop and horizontal flip, without changing the label.
    n = len(idx)
    y = rng.integers(0, 9, (n, 1, 1)) + np.arange(32)[None, :, None]
    x = np.arange(32)[None, None, :]
    x = np.where(rng.random((n, 1, 1)) < .5, 31 - x, x) + rng.integers(0, 9, (n, 1, 1))
    return padded[idx[:, None, None], y, x]


@torch.inference_mode()
def evaluate(model, x, y, batch, device):
    model.eval()
    all_logits = []
    for i in range(0, len(y), batch):
        all_logits.append(model(tensor_images(x[i:i+batch], device)).cpu())
    logits = torch.cat(all_logits)
    labels = torch.from_numpy(y.astype(np.int64))
    loss = nn.functional.cross_entropy(logits, labels).item()
    conf, pred = logits.softmax(1).max(1)
    return loss, (labels == pred).float().mean().item(), pred.numpy(), conf.numpy()


def train_group(name, data, args, cfg):
    x, y, train, vx, vy, padded = data
    aug, att = GROUPS[name]
    out = args.out / name
    out.mkdir(exist_ok=True)
    torch.manual_seed(args.seed)
    model = make_model(attention=att).to(args.device, memory_format=torch.contiguous_format)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr * .05)
    summary = dict(group=name, augmentation=aug, attention=att,
                   parameters=sum(p.numel() for p in model.parameters()), **cfg)
    history, best = [], -1.
    last = out/'latest.pt'
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=True)
        if saved['config'] != summary:
            raise ValueError('Existing run has incompatible configuration; select another --out.')
        history, best = saved['history'], saved['best']
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
    write_json(out/'config.json', summary)
    for ep in range(len(history)+1, args.epochs+1):
        t0 = time.perf_counter()
        lr = opt.param_groups[0]['lr']
        model.train()
        # Independent RNG streams: an augmentation draw can never alter the sample order.
        order = np.random.default_rng(args.seed + ep).permutation(train)
        rng = np.random.default_rng(args.seed + 10000 + ep)
        torch.manual_seed(args.seed + ep)
        sums = torch.zeros(2, device=args.device)
        for i in range(0, len(order), args.batch):
            idx = order[i:i+args.batch]
            imgs = augment_batch(padded, idx, rng) if aug else x[idx]
            batch_x = tensor_images(imgs, args.device)
            target = torch.from_numpy(y[idx]).to(args.device)
            opt.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = nn.functional.cross_entropy(logits, target)
            loss.backward()
            opt.step()
            with torch.no_grad():
                sums[0] += loss.detach()*len(idx)
                sums[1] += (logits.argmax(1) == target).sum()
        train_loss, train_acc = (sums/len(train)).cpu().tolist()
        val_loss, val_acc, _, _ = evaluate(model, vx, vy, args.batch, args.device)
        if not np.isfinite([train_loss, val_loss]).all():
            raise ValueError('Nonfinite metrics')
        row = dict(epoch=ep, lr=lr, train_loss=train_loss, train_acc=train_acc,
                   val_loss=val_loss, val_acc=val_acc, seconds=time.perf_counter()-t0)
        history.append(row)
        if val_acc > best:
            best = val_acc
            checkpoint(out/'best.pt', dict(epoch=ep, model=model.state_dict(), val_acc=val_acc))
        scheduler.step()
        checkpoint(last, dict(config=summary, model=model.state_dict(), optimizer=opt.state_dict(),
                              scheduler=scheduler.state_dict(), history=history, best=best))
        with (out/'history.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=row.keys()); w.writeheader(); w.writerows(history)
        print(f'{name} {ep}/{args.epochs} train={train_acc:.4f} val={val_acc:.4f} ({row["seconds"]:.1f}s)', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'data/cifar10.npz')
    parser.add_argument('--out', type=Path, default=ROOT/'results')
    parser.add_argument('--seed', type=int, default=SEED, help='Training randomness only; split is held fixed')
    parser.add_argument('--no-image-copy',action='store_true',help='Skip identical test-image file for replication runs')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=.002)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.data, allow_pickle=False) as ds:
        x, y = ds['x_train'], ds['y_train'].reshape(-1).astype(np.int64)
    train, val = split_indices(y)
    assert x.shape == (50000,32,32,3) and x.dtype == np.uint8 and len(np.unique(np.r_[train,val])) == 50000
    np.savez(args.out/'split.npz', train=train, val=val)
    cfg = dict(epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed, device=args.device,
               optimizer='AdamW', weight_decay=.01, scheduler='cosine', torch=str(torch.__version__),
               python=platform.python_version(), system=platform.platform(),
               split_sha256=hashlib.sha256(train.tobytes()+val.tobytes()).hexdigest(),
               model_sha256=hashlib.sha256((ROOT/'model.py').read_bytes()).hexdigest())
    pad = np.pad(x, ((0,0),(4,4),(4,4),(0,0)), mode='reflect')
    data = (x,y,train,x[val],y[val],pad)
    for name in GROUPS:
        train_group(name, data, args, cfg)
    # Hold the official test split out until all four validation-selected models are fixed.
    with np.load(args.data, allow_pickle=False) as ds:
        tx, ty = ds['x_test'], ds['y_test'].reshape(-1).astype(np.int64)
    assert tx.shape==(10000,32,32,3) and np.all(np.bincount(ty)==1000)
    if not args.no_image_copy:
        np.save(args.out/'test_images.npy',tx)
    for name, (_, att) in GROUPS.items():
        out = args.out/name
        ck = torch.load(out/'best.pt',map_location='cpu',weights_only=True)
        model = make_model(attention=att).to(args.device, memory_format=torch.contiguous_format)
        model.load_state_dict(ck['model'])
        loss, acc, pred, conf = evaluate(model, tx, ty, args.batch, args.device)
        np.savez_compressed(out/'predictions.npz', y_true=ty, y_pred=pred, confidence=conf)
        write_json(out/'test.json',dict(best_epoch=ck['epoch'],val_acc=ck['val_acc'],test_acc=acc,test_loss=loss))
        print(name, 'test', acc, flush=True)


if __name__ == '__main__':
    main()
