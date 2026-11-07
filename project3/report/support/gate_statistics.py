"""Measure learned gates in the primary validation-selected checkpoints without training."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from train import tensor_images, make_model

@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    with np.load(ROOT/'data/cifar10.npz',allow_pickle=False) as f:
        images=f['x_test']
    assert images.shape==(10000,32,32,3)
    data = dict(seed=20261107, split='official test, all 10000 images', mode='CPU eval',
                gate='2 * sigmoid(gate.expand output)', results={})
    for group in ('attention','both'):
        path=ROOT/'results'/group/'best.pt'
        state=torch.load(path,map_location='cpu',weights_only=True)
        model=make_model(True).eval()
        model.load_state_dict(state['model'])
        values=[]
        def capture(_module, _input, out):
            values.append((2*torch.sigmoid(out)).numpy())
        hook=model.gate.expand.register_forward_hook(capture)
        for start in range(0,len(images),128):
            model(tensor_images(images[start:start+128],'cpu'))
        hook.remove()
        g=np.vstack(values).astype(np.float64)
        assert g.shape==(10000,64)
        data['results'][group]=dict(epoch=state['epoch'], checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                 n_gates=int(g.size), mean=float(g.mean()), mean_abs_deviation_from_one=float(np.abs(g-1).mean()),
                                 percentiles_5_50_95=np.percentile(g,[5,50,95]).tolist(),
                                 mean_across_channel_of_per_channel_std_across_images=float(g.std(axis=0).mean()))
    out=Path(__file__).with_suffix('.json')
    out.write_text(json.dumps(data,indent=2)+'\n')
    print(out.read_text())

if __name__=='__main__':
    main()
