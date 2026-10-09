"""固定有限输入域与概率表：总体风险能精确求和，无测试估计误差。"""
from pathlib import Path  # 本地路径操作。
import numpy as np  # 数值数组与独立伪随机生成器。


def population():
    x = np.linspace(-1., 1., 21)  # 输入只取21个等概率格点。
    prob = np.where(x >= 0, .85, .15)  # 真标签为1的条件概率，保留标签噪声。
    thresholds = np.linspace(-1.1, 1.1, 23)  # 预先固定23个阈值，不从样本选择。
    predictions = (x[None, :] >= thresholds[:, None]).astype(int)  # [23,21]类别表。
    return dict(x=x, prob=prob, thresholds=thresholds, predictions=predictions)


def sample(n, seed):
    if not isinstance(n, (int, np.integer)) or n < 1:  # 拒绝无效样本量。
        raise ValueError('n must be a positive integer')
    pop = population()  # 同一预先固定总体。
    rng = np.random.default_rng(seed)  # 不污染全局随机状态。
    index = rng.integers(0, len(pop['x']), size=n)  # IID有放回抽取输入格点。
    y = (rng.random(n) < pop['prob'][index]).astype(int)  # 条件独立生成标签。
    return index, y  # 输入以0..20格点索引存储。


def generate(output=None):
    pop = population()  # 返回原创离散总体完整定义。
    index, y = sample(100, 7400)  # CSV样例固定100行。
    if output is not None:
        out = Path(output); out.mkdir(parents=True, exist_ok=True)
        np.savez(out/'population.npz', **pop)  # 可无损恢复总体。
        np.savetxt(out/'sample.csv', np.c_[index, pop['x'][index], y], delimiter=',',
                   header='grid_index,x,y', comments='')  # 业务字段在字典逐项解释。
    return pop


if __name__ == '__main__':
    generate(Path(__file__).resolve().parent/'data')  # 无网络访问。
