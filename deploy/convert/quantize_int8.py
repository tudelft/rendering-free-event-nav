"""Step 3: int8 post-training quantization of the fp32 actor ONNX.

QDQ format, per-channel weights, calibrated on the representative observations from make_calib.py.
The image input stays float (0..255); the (x-128)/32 affine that is the graph's first op folds
into conv0's int8 scale, so no separate uint8 input quantizer is needed. stedgeai is then invoked
with --input-data-type float32 --output-data-type float32 (see run_stedgeai.sh).

Usage:
  python deploy/convert/quantize_int8.py \
      --onnx deploy/convert/out/actor_fp32.onnx \
      --calib deploy/convert/out/calib.npz \
      --out deploy/convert/out/actor_int8.onnx
"""
import argparse
import numpy as np
import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat


class _Reader(CalibrationDataReader):
    def __init__(self, npz):
        d = np.load(npz)
        self.img = d['image'].astype(np.float32)          # [N,64,64,3] -> NCHW below
        self.vec = d['vector'].astype(np.float32)
        self.i = 0

    def get_next(self):
        if self.i >= len(self.img):
            return None
        x = self.img[self.i].transpose(2, 0, 1)[None]     # NHWC sample -> [1,3,64,64]
        v = self.vec[self.i][None]
        self.i += 1
        return {'image': x, 'vector': v}


def _strip_allowzero(path):
    g = onnx.load(path)
    for n in g.graph.node:
        if n.op_type == 'Reshape':
            keep = [a for a in n.attribute if a.name != 'allowzero']
            del n.attribute[:]; n.attribute.extend(keep)
    onnx.save(g, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', default='deploy/convert/out/actor_fp32.onnx')
    ap.add_argument('--calib', default='deploy/convert/out/calib.npz')
    ap.add_argument('--out', default='deploy/convert/out/actor_int8.onnx')
    A = ap.parse_args()
    quantize_static(
        A.onnx, A.out, _Reader(A.calib),
        quant_format=QuantFormat.QDQ, per_channel=True,
        weight_type=QuantType.QInt8, activation_type=QuantType.QInt8)
    _strip_allowzero(A.out)          # quantizer can re-add Reshape allowzero; stedgeai v4 rejects it
    print('wrote', A.out)


if __name__ == '__main__':
    main()
