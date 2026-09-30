"""验证 quant_cuda 3-bit kernel 算得对不对。

思路：同一份权重走两条路，比较结果
  路径 A（模拟）：用纯 PyTorch 的 quantize() 反量化回 FP32，再做普通 matmul
  路径 B（真 kernel）：权重打包成 3-bit 整数，交给 CUDA kernel 做 matmul
两条路的数值应该几乎一致。
"""
import torch
import torch.nn as nn

import quant_cuda
from quant import Quantizer, quantize, Quant3Linear

torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)

DEV = torch.device('cuda:0')
M, N = 4096, 4096  # in_features, out_features

layer = nn.Linear(M, N, bias=True)
vec = torch.randn(M, device=DEV)

# 3-bit 量化，非对称
q = Quantizer()
q.configure(3, perchannel=True, sym=False, mse=False)
q.find_params(layer.weight.data, weight=True)
layer.weight.data = quantize(layer.weight.data, q.scale, q.zero, q.maxq)

# 打包成 3-bit 整数权重
qlayer = Quant3Linear(layer.in_features, layer.out_features)
qlayer.pack(layer, q.scale, q.zero)
qlayer = qlayer.to(DEV)
layer = layer.to(DEV)

with torch.no_grad():
    simu = layer(vec)                      # 路径 A
    kern = qlayer(vec)                     # 路径 B (fp32)
    qlayer.faster = True
    kern_h = qlayer(vec.half())            # 路径 B (half, faster kernel)

    print('simu[:4]      :', simu[:4].tolist())
    print('kern[:4]      :', kern[:4].tolist())
    print()
    print('fp32 kernel  最大绝对误差 :', (simu - kern).abs().max().item())
    print('fp32 kernel  相对误差     :', ((simu - kern).abs().max() / simu.abs().max()).item())
    print('half kernel  最大绝对误差 :', (simu.half() - kern_h).abs().max().item())
    print()
    ok = (simu - kern).abs().max().item() < 1e-2
    print('结果:', 'PASS —— kernel 与模拟量化一致' if ok else 'FAIL —— 数值对不上')
