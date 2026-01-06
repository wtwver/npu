#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from rknn.api import RKNN


class Mt19937:
    def __init__(self, seed: int):
        self.mt = [0] * 624
        self.index = 624
        self.mt[0] = seed & 0xFFFFFFFF
        for i in range(1, 624):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & 0xFFFFFFFF

    def extract(self) -> int:
        mag01 = [0, 0x9908B0DF]
        if self.index >= 624:
            for kk in range(624 - 397):
                y = (self.mt[kk] & 0x80000000) | (self.mt[kk + 1] & 0x7FFFFFFF)
                self.mt[kk] = self.mt[kk + 397] ^ (y >> 1) ^ mag01[y & 1]
            for kk in range(624 - 397, 623):
                y = (self.mt[kk] & 0x80000000) | (self.mt[kk + 1] & 0x7FFFFFFF)
                self.mt[kk] = self.mt[kk - (624 - 397)] ^ (y >> 1) ^ mag01[y & 1]
            y = (self.mt[623] & 0x80000000) | (self.mt[0] & 0x7FFFFFFF)
            self.mt[623] = self.mt[396] ^ (y >> 1) ^ mag01[y & 1]
            self.index = 0

        y = self.mt[self.index]
        self.index += 1
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= (y >> 18)
        return y & 0xFFFFFFFF

    def uniform(self, low: float, high: float) -> float:
        a = self.extract() >> 5
        b = self.extract() >> 6
        rnd = (a * 67108864.0 + b) / 9007199254740992.0
        return low + (high - low) * rnd


def generate_weights(in_ch: int, out_ch: int, h: int, w: int,
                     kh: int, kw: int, low: float, high: float):
    rng = Mt19937(0)
    # Consume input RNG draws (matches conv2d_multi.cpp)
    for _ in range(1):  # batch
        for _ in range(in_ch):
            for _ in range(h):
                for _ in range(w):
                    rng.uniform(low, high)
    weights = []
    for _oc in range(out_ch):
        for _ic in range(in_ch):
            for _kh in range(kh):
                for _kw in range(kw):
                    weights.append(rng.uniform(low, high))
    return weights


def build_model(size: int, out_dir: Path, force: bool) -> None:
    in_ch = 3
    out_ch = 6
    kh = kw = 1
    model_name = f"conv2d_1x1_{size}"
    onnx_path = out_dir / f"{model_name}.onnx"
    rknn_path = out_dir / f"{model_name}.rknn"

    if not force and rknn_path.exists():
        print(f"[skip] {rknn_path}")
        return

    weights = generate_weights(in_ch, out_ch, size, size, kh, kw, -2.0, 2.0)
    weight_tensor = torch.tensor(weights, dtype=torch.float32).reshape(out_ch, in_ch, kh, kw)

    class SimpleConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                                        kernel_size=(kh, kw), bias=False)
            with torch.no_grad():
                self.conv.weight.copy_(weight_tensor)

        def forward(self, x):
            return self.conv(x)

    model = SimpleConv()
    dummy = torch.zeros(1, in_ch, size, size, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=False,
        input_names=['input'],
        output_names=['output'],
    )

    rknn = RKNN()
    rknn.config(target_platform='rk3588')
    ret = rknn.load_onnx(model=str(onnx_path), input_size_list=[[1, in_ch, size, size]])
    if ret != 0:
        raise RuntimeError(f"load_onnx failed for {onnx_path}: {ret}")
    ret = rknn.build(do_quantization=False, dataset=None)
    if ret != 0:
        raise RuntimeError(f"build failed for {onnx_path}: {ret}")
    ret = rknn.export_rknn(str(rknn_path))
    if ret != 0:
        raise RuntimeError(f"export_rknn failed for {rknn_path}: {ret}")
    rknn.release()
    print(f"[ok] {rknn_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=1)
    parser.add_argument("--out-dir", default="ops_rknn/models")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = max(1, args.start)
    end = max(start, args.end)
    for n in range(start, end + 1):
        build_model(n, out_dir, args.force)


if __name__ == "__main__":
    main()
