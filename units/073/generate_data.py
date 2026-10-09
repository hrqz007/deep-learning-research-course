"""原创可分二维数据与凸二次目标；无下载、无外部状态。"""
from pathlib import Path  # Path把路径连接与操作系统分隔符分开。
import numpy as np  # 数组按行保存样本、按列保存特征。


def generate(output=None):
    """生成固定小数据；保留显式支持向量便于手算最大间隔。"""
    z = np.array([[1., 0.], [0., 2.], [1., 1.], [2., 1.], [1.5, 2.]])  # 正类。
    x = np.vstack([z, -z])  # 负类是正类镜像，边界过原点。
    y = np.r_[np.ones(len(z)), -np.ones(len(z))]  # 标签是正1和负1。
    q = np.diag([1., 4.])  # 二次目标曲率；最大特征值L=4。
    w0 = np.array([4., 2.])  # 初始点；最优点为[0,0]。
    result = dict(x=x, y=y, q=q, w0=w0)  # 字段与数据字典一致。
    if output is not None:  # 允许Notebook仅返回数组、不写盘。
        folder = Path(output)  # 接受字符串或Path对象。
        folder.mkdir(parents=True, exist_ok=True)  # 自动建立数据目录。
        np.savez(folder / 'data.npz', **result)  # 数值数组，无pickle对象。
        np.savetxt(folder / 'separable.csv', np.c_[x, y], delimiter=',',
                   header='x0,x1,y', comments='')  # 便于不用Python也能检查。
    return result  # 所有实验使用同一组可核验原始数组。


if __name__ == '__main__':
    generate(Path(__file__).resolve().parent / 'data')  # 只写本单元数据。
