# 这个代码是量化的核心代码，需要注意的是量化代码与GPTQ代码本质上不是一样的，这个部分是最底层的进行量化的操作

import torch
import torch.nn as nn

try:
    import quant_cuda
except:
    print('CUDA extension not installed.')

# 自带的C++/CUDA 扩展，需要在机器上自己进行编译


def quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

# 量化的基本的函数，用来执行量化逻辑，但是最后返回的是反量化之后的值，本质上是一个fake量化的操作

class Quantizer(nn.Module):
    # 执行量化操作的类

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))
        # 初始化三个需要用到的参数，我们使用zero-point量化，需要用到的就是上面函数需要传入的三个重要的参数
        # 使用buffer是因为这些参数不参加梯度训练

    def configure(
        # 这个configure函数不是一个魔法函数(类似于forward,_apply,generate等函数，这个函数就是一个普通的方法，是自己手动调，自己手动写的)
        # 这个函数只是把所有的配置都记下来，但是并没有做真正的计算
            self,
            bits, perchannel=False, sym=True,
            mse=False, norm=2.4, grid=100, maxshrink=.8
        ):
        # bits是量化的bit量，perchannel就是量化的粒度，sym就是对称量化，如果是对称量化的话，零点就会固定在区间的中点(maxq + 1)/2 (加1 是因为有0), mse表示是否放弃朴素的min-max，改用网格搜索来寻找让量化误差最小的s,z。后面的都是mse的参数
        # for i in range(int(maxshrink * grid)):
        #     p = 1 - i / grid  p从0.21到1,步长为0.1
        #     xmin1 = p * xmin
        #     xmax1 = p * xmax
        # grid 就是搜索的分辨率，分别控制步长和循环次数，grid越大，搜索的就越仔细，但是会更慢，而且是纯python的，每次迭代都会对整个权重矩阵进行一次量化和求幂，会非常慢
        # maxshrink 是搜索的下界，也就是搜索的区间最后会被压缩
        # q -= x
        # q.abs_()
        # q.pow_(self.norm) norm参数是平方的参数
        # err = torch.sum(q, 1) 每个输出通道都有一个误差值，最后想要使得这个误差值最小
        # 指数略大于2，会让大误差更加明显，是工程经验得出的结果
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink

    def find_params(self, x, weight=False):
        # weight参数是告诉函数，我们传入的tensor 是权重还是激活值，关键是因为权重和激活值本身的形状是不一样的
        # 这个函数本身不改变任何的权重，而是写出scale 和 zero两个数
        dev = x.device
        self.maxq = self.maxq.to(dev)
        # 搬到同一个设备，因为权重是在GPU上面的，但是configure的参数是在CPU上面的，所以需要进行数据的搬运

        shape = x.shape
        # 保存原来的形状，下面的部分是重排形状的部分
        if self.perchannel:
            # 如果进行perchannel粒度的计算
            if weight:
                x = x.flatten(1)
                # tensor.flatten(start_dim, end_dim)就是从start_dim维度开始合并到end_dim的维度，在这里就是从dim = 1开始一路合并到最后一维，对于nn.Linear()来说，实际上就是没有什么变化，但是对于Conv2d(out, in, kh, kw)来说就会把后面的所有的权重乘起来
                # 但是本质上对dim = 0 是没有改变的，而dim = 0也就是输出通道
            else:
                if len(shape) == 4:
                    # 4D的情况是卷积激活值[N, C, H, W]，N表示图片的数量，C是图片特征的数量，H,W是图片的长和宽
                    x = x.permute([1, 0, 2, 3])
                    # permute函数是将不同的维度重新进行排序，但是本质上的内存还是没有改变，也就是完成了一个视图上的转化，内存上没有开销
                    x = x.flatten(1)
                if len(shape) == 3:
                    # 这个就是常规的激活值[B, T, C]
                    x = x.reshape((-1, shape[-1])).t()
                    # reshape 之后变成了[B x T, C]
                    # t()就是2Dtensor的转置，最后就变成了[C, B x T]
                if len(shape) == 2:
                    # 2D就是全连接激活值[N, C],我们只需要进行转置就可以了
                    x = x.t()
            # 本质上都统一成输出通道上面的全部的值，这样我们才可以得到每一个通道的min/max
        else:
            x = x.flatten().unsqueeze(0)
            # tensor.unsqueeze(dim)就是在dim的位置插入一个长度为1的新的维度，就类似于我们把最后的输出通道变成了1，然后后面的整个tensor都是这个输出通道的维度

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        # tensor.min/max(dim) 一旦带上dim，返回的就是一个元组[0]是最小值/最大值[1]是所取的数字的位置，不带参数的时候就会返回一个全局最小的值，这个时候就不是一个元组，而就是一个值了
        # torch.minium(a, b)是两个tensor逐位置进行比大小，因为最后两个shape都是输出通道的大小，所以可以直接进行比较，这样是防止最小的数字大于0
        xmax = torch.maximum(x.max(1)[0], tmp)
        # 这个步骤是防止最大的数字小于0
        # 为的是把量化区间强行撑到包含0

        # 接下来就是进行算法上的操作了

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            # self.scale = self.scale.unsqueeze(1)
            # self.zero = self.zero.unsqueeze(1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

class ActQuantWrapper(nn.Module):

    def __init__(self, module):
        super(ActQuantWrapper, self).__init__()
        self.module = module
        shape = [1] * len(self.module.weight.shape)
        if len(shape) == 4:
            shape[1] = self.module.weight.shape[1]
        if len(shape) == 3:
            shape[2] = self.module.weight.shape[2]
        if len(shape) == 2:
            shape[1] = self.module.weight.shape[1]
        self.quantizer = Quantizer(shape=shape)

    def forward(self, x):
        return self.module(self.quantizer.quantize(x))

def add_actquant(module, name='', layers=[nn.Conv2d, nn.Linear]):
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp))
        if type(tmp) == nn.Sequential:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, nn.Sequential(*replaced))
        if type(tmp) == torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, nn.ModuleList(replaced))
    for name1, child in module.named_children():
        add_actquant(child, name + '.' + name1 if name != '' else name1, layers)

import time

class Quant4Linear(nn.Module):

    def __init__(self, linear, scales, zeros):
        super().__init__()
        self.register_buffer('zeros', zeros.clone() * scales)
        self.register_buffer('scales', scales)
        self.register_buffer('bias', linear.bias.data)
        intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(torch.int)
        intweight = intweight.t().contiguous()
        self.register_buffer('qweight', torch.zeros(
            (intweight.shape[0] // 8, intweight.shape[1]), dtype=torch.int, device=self.bias.device
        ))
        for i in range(intweight.shape[0]):
            self.qweight[i // 8] |= intweight[i] << (4 * (i % 8))
        # self.linear = linear.to(torch.device('cuda:0'))

    def forward(self, x):
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            y = self.bias.clone()
            outshape[-1] = self.bias.numel()
            quant_cuda.vecquant4matmul(x, self.qweight, y, self.scales, self.zeros)
            # y = self.linear(x)
            return y.reshape(outshape)
        print(x.shape)
        raise ValueError('Only supports a single token currently.')

def make_quant4(module, quantizers, name=''):
    if isinstance(module, Quant4Linear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in quantizers:
            setattr(
                module, attr,
                Quant4Linear(tmp, quantizers[name1].scale, quantizers[name1].zero)
            )
    for name1, child in module.named_children():
        make_quant4(child, quantizers, name + '.' + name1 if name != '' else name1)
