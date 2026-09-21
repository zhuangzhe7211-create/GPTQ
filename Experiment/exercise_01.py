"""第一份作业：两处 TODO。只依赖 Python 标准库。

本练习的 candidate 是一个标量权重，输入 x 是各个样本的标量输入。
它用于理解加权目标，不是 GPTQ/ResComp 的实现。
"""


def weighted_error(pred, target, weights):
    """返回 0.5 * sum_t weights[t] * (pred[t]-target[t])**2。

    输入是等长非空数值列表，weights 为非负数。
    不做 sum(weights) 归一化；先严格实现数学定义。
    """
    raise NotImplementedError("由你实现加权平方误差")


def best_candidate(candidates, x, target, weights):
    """枚举给定量化候选，返回使 weighted_error 最小的标量权重。

    对候选 q，pred[t] = q*x[t]。并列最优时返回 candidates 中最先出现的。
    调用上面的 weighted_error，不能根据测试样例硬编码答案。
    """
    raise NotImplementedError("由你实现有限量化网格搜索")
