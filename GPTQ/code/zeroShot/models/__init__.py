# 这个文件的本质是一个根据模型的名称选择OPT或者BLOOM模型的文件

from . import opt
from . import bloom

MODEL_REGISTRY = {
    'opt': opt.OPT,
    'bloom': bloom.BLOOM
}


def get_model(model_name):
    if 'opt' in model_name:
        return MODEL_REGISTRY['opt']
    elif 'bloom' in model_name:
        return MODEL_REGISTRY['bloom']
    return MODEL_REGISTRY[model_name]

# 最后返回的是一个模型，接下来我们分别来看一下两个模型是怎么写的
