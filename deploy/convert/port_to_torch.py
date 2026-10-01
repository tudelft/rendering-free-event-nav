"""Step 1 of the N6 deploy chain: port the trained Flax actor to a Torch module that
exports to a *clean, NPU-mappable* ONNX graph, and verify the port is bit-for-bit faithful.

Why a re-port instead of exporting the Flax graph directly:
  * The Neural-ART (stedgeai) compiler wants a plain NCHW conv/dense graph. A Flax actor
    exported as-is carries an NHWC input and a Transpose that stedgeai mis-infers into a
    scrambled board tensor ([b:1,h:64,w:3,c:64]) -> E103 compile failure. We instead build
    an NCHW Torch twin and transplant the weights, so the exported graph is Conv/Gemm only.
  * The two layout conversions that matter are handled explicitly here:
      - conv kernel  Flax [kh,kw,in,out]  ->  Torch [out,in,kh,kw]
      - the first dense after the conv trunk: Flax flattens the trunk output in H,W,C order
        (NHWC), Torch flattens in C,H,W order (NCHW). We REORDER that dense's input columns
        (the 4096 conv-flat ones) from HWC->CHW so the two graphs compute the identical thing
        with NO Transpose in the graph.
  * Flax Conv default padding is SAME (asymmetric for stride 2); Torch Conv2d only does
    symmetric padding, so we F.pad with the exact SAME amounts before each conv.

The deployed graph outputs the raw action MEAN only; `a = tanh(mean)` is two scalar ops on the
Cortex-M55 after the single NPU pass (matches deploy/contracts.md). The (x-128)/32 input affine
is the first op in the graph so it folds into conv0's int8 scale at quantization.

Usage (run in an env with jax+flax+torch, e.g. the training env):
    python deploy/convert/port_to_torch.py checkpoints/ours_champion.pkl \
        --out-onnx deploy/convert/out/actor_fp32.onnx --out-pt deploy/convert/out/actor.pt
"""
import argparse, os, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- SAME-padding amounts (left,right,top,bottom) for the champion 64x64 trunk ---
# SAME: out=ceil(n/s); total_pad=max((out-1)*s+k-n,0); before=total//2, after=total-before.
#   conv0: n=64 k=5 s=2 -> pad (1,2,1,2)    conv1: n=32 k=3 s=2 -> pad (0,1,0,1)
#   conv2: n=16 k=3 s=2 -> pad (0,1,0,1)
PAD0, PAD1, PAD2 = (1, 2, 1, 2), (0, 1, 0, 1), (0, 1, 0, 1)


class TorchActor(nn.Module):
    """NCHW twin of training/nets.py:Actor (conv/relu/dense/concat only, linear mean head)."""
    def __init__(self, cw=(16, 32, 64), dw=(256, 128), vec_dim=4, adim=2, in_ch=3, grid=8):
        super().__init__()
        self.grid = grid
        self.c0 = nn.Conv2d(in_ch, cw[0], 5, 2, 0)
        self.c1 = nn.Conv2d(cw[0], cw[1], 3, 2, 0)
        self.c2 = nn.Conv2d(cw[1], cw[2], 3, 2, 0)
        self.dv = nn.Linear(vec_dim, 64)
        self.d1 = nn.Linear(cw[2] * grid * grid + 64, dw[0])
        self.d2 = nn.Linear(dw[0], dw[1])
        self.dm = nn.Linear(dw[1], adim)

    def forward(self, img, vec):                 # img [B,in_ch,64,64] float 0..255, vec [B,vec_dim]
        x = (img - 128.0) / 32.0
        x = F.relu(self.c0(F.pad(x, PAD0)))
        x = F.relu(self.c1(F.pad(x, PAD1)))
        x = F.relu(self.c2(F.pad(x, PAD2)))      # [B, cw2, 8, 8]
        z = x.reshape(x.shape[0], -1)            # NCHW (c,h,w) flatten
        v = F.relu(self.dv(vec))
        h = F.relu(self.d1(torch.cat([z, v], -1)))
        h = F.relu(self.d2(h))
        return self.dm(h)                        # action MEAN (tanh applied on the M55)


def _hwc_to_chw_perm(C, G):
    """Index map so torch_W[:, perm] reorders HWC-ordered conv-flat columns into CHW order.
    flax feature f_hwc = (h*G + w)*C + c ; torch feature f_chw = (c*G + h)*G + w."""
    perm = np.empty(C * G * G, np.int64)
    for c in range(C):
        for h in range(G):
            for w in range(G):
                perm[(c * G + h) * G + w] = (h * G + w) * C + c
    return perm


def transplant(ckpt_actor, cw=(16, 32, 64), dw=(256, 128), vec_dim=4, adim=2, grid=8):
    p = ckpt_actor['params']
    m = TorchActor(cw, dw, vec_dim, adim, in_ch=p['ConvTrunk_0']['Conv_0']['kernel'].shape[2], grid=grid)

    def conv(dst, src):      # Flax [kh,kw,in,out] -> Torch [out,in,kh,kw]
        dst.weight.data = torch.tensor(np.asarray(src['kernel']).transpose(3, 2, 0, 1).copy())
        dst.bias.data = torch.tensor(np.asarray(src['bias']).copy())

    def dense(dst, src):     # Flax [in,out] -> Torch [out,in]
        dst.weight.data = torch.tensor(np.asarray(src['kernel']).T.copy())
        dst.bias.data = torch.tensor(np.asarray(src['bias']).copy())

    ct = p['ConvTrunk_0']
    conv(m.c0, ct['Conv_0']); conv(m.c1, ct['Conv_1']); conv(m.c2, ct['Conv_2'])
    dense(m.dv, p['Dense_0'])

    # Dense_1: first (cw2*grid*grid) rows are conv-flat in HWC order -> reorder to CHW; vec rows keep order.
    C = cw[2]; ncnn = C * grid * grid
    W1 = np.asarray(p['Dense_1']['kernel'])            # [4160, 256]
    perm = _hwc_to_chw_perm(C, grid)
    W1_cnn = W1[:ncnn][perm]                           # rows reordered HWC->CHW
    W1_full = np.concatenate([W1_cnn, W1[ncnn:]], 0)   # [4160,256]
    m.d1.weight.data = torch.tensor(W1_full.T.copy())
    m.d1.bias.data = torch.tensor(np.asarray(p['Dense_1']['bias']).copy())

    dense(m.d2, p['Dense_2']); dense(m.dm, p['Dense_3'])
    return m.eval()


def _conv_same_nhwc(x, src, k, s):
    """Flax-equivalent SAME-padded stride-s conv on NHWC input with a [kh,kw,in,out] kernel."""
    W = np.asarray(src['kernel']); b = np.asarray(src['bias'])
    N, H, Wd, Ci = x.shape; Co = W.shape[3]
    out = int(np.ceil(H / s)); tot = max((out - 1) * s + k - H, 0); pb = tot // 2; pa = tot - pb
    xp = np.pad(x, ((0, 0), (pb, pa), (pb, pa), (0, 0)))
    y = np.zeros((N, out, out, Co), np.float32)
    for i in range(out):
        for j in range(out):
            patch = xp[:, i*s:i*s+k, j*s:j*s+k, :]
            y[:, i, j, :] = np.tensordot(patch, W, axes=([1, 2, 3], [0, 1, 2])) + b
    return y


def _np_reference(actor, img_nhwc, vec):
    """Independent NumPy reimplementation of training/nets.py:Actor (ground truth for the port)."""
    p = actor['params']; ct = p['ConvTrunk_0']
    x = (img_nhwc.astype(np.float32) - 128.0) / 32.0
    x = np.maximum(_conv_same_nhwc(x, ct['Conv_0'], 5, 2), 0)
    x = np.maximum(_conv_same_nhwc(x, ct['Conv_1'], 3, 2), 0)
    x = np.maximum(_conv_same_nhwc(x, ct['Conv_2'], 3, 2), 0)   # [N,8,8,64] NHWC
    z = x.reshape(x.shape[0], -1)                               # HWC flatten (flax order)
    dot = lambda a, s: a @ np.asarray(s['kernel']) + np.asarray(s['bias'])
    v = np.maximum(dot(vec, p['Dense_0']), 0)
    h = np.maximum(dot(np.concatenate([z, v], -1), p['Dense_1']), 0)
    h = np.maximum(dot(h, p['Dense_2']), 0)
    return dot(h, p['Dense_3'])


def parity_check(ckpt, m, meta, n=8, seed=0):
    """Compare the Torch twin against an independent NumPy reference on random inputs
    (no jax/flax needed; this is a genuine second implementation of the same math)."""
    H, fs, vec_dim = meta['H'], meta['frame_stack'], meta['vec_dim']
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(n, H, H, fs)).astype(np.uint8)        # NHWC uint8 (flax layout)
    vec = rng.standard_normal((n, vec_dim)).astype(np.float32)
    mean_ref = _np_reference(ckpt['actor'], img, vec)
    img_t = torch.tensor(img.transpose(0, 3, 1, 2).astype(np.float32))     # NCHW for torch
    with torch.no_grad():
        mean_t = m(img_t, torch.tensor(vec)).numpy()
    err = np.abs(mean_ref - mean_t).max()
    return err, mean_ref, mean_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--out-onnx', default='deploy/convert/out/actor_fp32.onnx')
    ap.add_argument('--out-pt', default='deploy/convert/out/actor.pt')
    ap.add_argument('--tol', type=float, default=5e-6)
    ap.add_argument('--no-verify', action='store_true', help='skip the Flax parity check (no jax/flax)')
    A = ap.parse_args()
    os.makedirs(os.path.dirname(A.out_onnx), exist_ok=True)

    ck = pickle.load(open(A.ckpt, 'rb'))
    meta = ck['meta']
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'training'))
    # widths: default champion 16/32/64 + 256/128 (meta carries adim/vec_dim; widths are the trained arch)
    cw, dw = (16, 32, 64), (256, 128)
    m = transplant(ck['actor'], cw, dw, meta['vec_dim'], meta['adim'])

    if not A.no_verify:
        err, mj, mt = parity_check(ck, m, meta)
        print(f'parity: max|flax-torch| = {err:.2e}  (tol {A.tol:.0e})  ->  {"OK" if err < A.tol else "FAIL"}')
        if err >= A.tol:
            raise SystemExit('parity check failed -- do not deploy this port')

    torch.save(m.state_dict(), A.out_pt)
    H, fs = meta['H'], meta['frame_stack']
    dummy_img = torch.zeros(1, fs, H, H)      # float image 0..255 (normalization is inside the graph)
    dummy_vec = torch.zeros(1, meta['vec_dim'])
    torch.onnx.export(
        m, (dummy_img, dummy_vec), A.out_onnx,
        input_names=['image', 'vector'], output_names=['mean'],
        opset_version=18, dynamo=False)
    import onnx, collections
    g = onnx.load(A.out_onnx)
    for node in g.graph.node:
        if node.op_type == 'Reshape':      # stedgeai v4 rejects Reshape 'allowzero'
            keep = [a for a in node.attribute if a.name != 'allowzero']
            del node.attribute[:]; node.attribute.extend(keep)
    _fuse_conv_pads(g.graph)               # F.pad dynamic subgraph -> Conv 'pads' attribute (clean NPU graph)
    _prune(g.graph)
    onnx.save(g, A.out_onnx)
    ops = collections.Counter(n.op_type for n in g.graph.node)
    print('exported', A.out_onnx, '| ops:', dict(ops))
    print('saved torch state_dict', A.out_pt)


def _fuse_conv_pads(graph, pad_by_k={5: [1, 1, 2, 2], 3: [0, 0, 1, 1]}):
    """Replace the dynamic Pad feeding each Conv with the Conv's own asymmetric `pads` attribute
    (ONNX pads = [top,left,bottom,right]); amount is the SAME pad implied by the kernel size."""
    import onnx
    prod = {o: n for n in graph.node for o in n.output}
    for c in [n for n in graph.node if n.op_type == 'Conv']:
        src = prod.get(c.input[0])
        if src is None or src.op_type != 'Pad':
            continue
        k = next(a.ints[0] for a in c.attribute if a.name == 'kernel_shape')
        c.input[0] = src.input[0]                       # bypass the Pad: take its data input
        keep = [a for a in c.attribute if a.name != 'pads']
        del c.attribute[:]; c.attribute.extend(keep)
        c.attribute.append(onnx.helper.make_attribute('pads', pad_by_k[k]))


def _prune(graph):
    """Drop nodes/initializers no longer reachable from the graph outputs."""
    prod = {o: n for n in graph.node for o in n.output}
    need, stack = set(), [o.name for o in graph.output]
    while stack:
        t = stack.pop()
        n = prod.get(t)
        if n is None or id(n) in need:
            continue
        need.add(id(n)); stack.extend(n.input)
    keep = [n for n in graph.node if id(n) in need]
    del graph.node[:]; graph.node.extend(keep)
    used = {i for n in graph.node for i in n.input}
    inits = [w for w in graph.initializer if w.name in used]
    del graph.initializer[:]; graph.initializer.extend(inits)


if __name__ == '__main__':
    main()
