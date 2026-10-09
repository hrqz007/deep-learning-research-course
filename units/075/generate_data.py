"""原创无噪声一维函数；训练点与插值网格固定且角色分开。"""
from pathlib import Path  # 保存本地数组。
import numpy as np  # 确定性解析函数，无下载。


def target(x):
    return np.sin(2.4*x)+.25*np.cos(5*x)  # 每个x对应唯一目标值。


def generate(output=None):
    x=np.linspace(-1.5,1.5,24)  # 24个固定训练点，无IID统计声明。
    grid=np.linspace(-1.5,1.5,151)  # 仅评价插值行为，不外推至区间外。
    data=dict(x=x,y=target(x),grid=grid,grid_y=target(grid))  # 字段明确。
    if output is not None:
        out=Path(output);out.mkdir(parents=True,exist_ok=True)
        np.savez(out/'data.npz',**data)  # 所有数组均float64。
        np.savetxt(out/'train.csv',np.c_[x,data['y']],delimiter=',',header='x,y',comments='')
    return data


if __name__=='__main__':
    generate(Path(__file__).resolve().parent/'data')
