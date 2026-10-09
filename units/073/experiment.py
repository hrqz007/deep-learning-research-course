"""用真实梯度下降检查收敛界和隐式选解；全CPU、离线、float64。"""
from pathlib import Path  # 管理输出目录。
import argparse,json,platform,time  # CLI、机器可读结果、版本与墙钟计时。
import numpy as np  # 数组和稳定的数值运算。
from scipy.special import expit  # 稳定计算sigmoid，避免指数溢出。
from generate_data import generate  # 同目录的原创数据生成器。


def logistic(w, x, y):
    """返回平均logistic损失及解析梯度；y只取正负1。"""
    margins = y * (x @ w)  # @为矩阵乘向量：每个样本的带符号分数。
    loss = np.logaddexp(0., -margins).mean()  # log(1+exp(-margin))稳定形式。
    grad = -(x.T @ (y * expit(-margins))) / len(y)  # 链式法则，除样本数。
    return float(loss), grad  # Python标量与d维梯度。


def descent_quadratic(q, w0, eta, steps):
    """最小化f(w)=w^T Q w/2；Q在实验中对称正定。"""
    w = np.asarray(w0, dtype=float).copy()  # 副本防止改动调用者初值。
    points = [w.copy()]  # 保留t=0，避免把步数错位。
    for _ in range(steps):  # 每次循环是一轮全量更新。
        w -= eta * (q @ w)  # 目标梯度为Qw。
        points.append(w.copy())  # copy避免所有历史指向最后一帧。
    points = np.asarray(points)  # 形状为[steps+1, d]。
    losses = np.einsum('ti,ij,tj->t', points, q, points) / 2  # 逐帧二次型。
    return points, losses  # 不把理论上界混入实测损失。


def train_logistic(x, y, steps=20000):
    """预定三种初始化，固定eta=1/L，保存完整轨迹。"""
    smooth = float(np.linalg.eigvalsh(x.T @ x / len(y)).max() / 4)  # 全局L上界。
    eta = 1 / smooth  # 在平滑目标的保守稳定范围内。
    reference = np.array([1., .5])  # 两条约束w0>=1、2w1>=1同时取等。
    ref_dir = reference / np.linalg.norm(reference)  # 硬间隔方向。
    runs = []  # 三种初始化都报告，不挑最佳。
    arrays = {}  # 轨迹存NPZ避免在JSON中塞入过长数组。
    for run_id, initial in enumerate([[0., 0.], [3., -2.], [-2., 3.]]):
        w = np.array(initial, dtype=float)  # 创建独立初始参数。
        record = np.empty((steps + 1, 5))  # 列：损失、范数、角度、间隔、错误率。
        weights = np.empty((steps + 1, 2))  # 保留完整w，允许独立重算。
        for t in range(steps + 1):
            loss, grad = logistic(w, x, y)  # 更新前记录第t帧。
            norm = np.linalg.norm(w)  # 欧氏长度。
            cosine = np.dot(w, ref_dir) / norm if norm else 0.  # 零向量无方向。
            angle = np.degrees(np.arccos(np.clip(cosine, -1, 1))) if norm else np.nan
            margin = np.min(y * (x @ w)) / norm if norm else np.nan  # 几何间隔。
            record[t] = [loss, norm, angle, margin, np.mean(y * (x @ w) <= 0)]
            weights[t] = w  # 写入数值，自动复制到历史数组。
            if t < steps:  # 最后一帧之后不再进行隐藏更新。
                w -= eta * grad  # 全批梯度下降，无显式惩罚项。
        arrays[f'run{run_id}_metrics'] = record  # NaN仅用于零向量未定义量。
        arrays[f'run{run_id}_weights'] = weights
        runs.append(dict(initial=initial, final_loss=record[-1, 0], final_norm=record[-1, 1],
                         final_angle_deg=record[-1, 2], final_margin=record[-1, 3],
                         final_error=record[-1, 4], angle_at_10=float(record[10, 2])))
    return dict(L=smooth, eta=eta, steps=steps, hard_margin_w=reference.tolist(),
                hard_margin_geometric=float(1 / np.linalg.norm(reference)), runs=runs), arrays


def run(output='outputs'):
    """保存每个预定实验，返回摘要；输出目录由调用者指定。"""
    start = time.perf_counter()  # 计时覆盖数据、训练和结果写入之前的计算。
    out = Path(output); out.mkdir(parents=True, exist_ok=True)  # 不修改源码。
    data = generate()  # 数据可以完全离线重新产生。
    good, loss = descent_quadratic(data['q'], data['w0'], .25, 50)  # eta=1/L。
    bad, bad_loss = descent_quadratic(data['q'], data['w0'], .6, 25)  # 超过2/L。
    t = np.arange(1, len(loss))  # t=0时1/t界没有定义。
    bound = 4 * np.dot(data['w0'], data['w0']) / (2 * t)  # L D^2/(2t)。
    implicit, arrays = train_logistic(data['x'], data['y'])  # 真实训练三条轨迹。
    arrays.update(quad_points=good, quad_loss=loss, quad_bound=bound,
                  bad_points=bad, bad_loss=bad_loss, **data)  # 保存原始输入便于核验。
    np.savez_compressed(out / 'trajectories.npz', **arrays)  # 无模型pickle。
    summary = dict(unit='073', catalog_id='T1', quadratic=dict(L=4., eta=.25,
                   max_bound_excess=float(np.max(loss[1:] - bound)), final_loss=float(loss[-1]),
                   bad_eta=.6, bad_final_loss=float(bad_loss[-1])), logistic=implicit,
                   runtime=dict(python=platform.python_version(), numpy=np.__version__,
                   seconds=time.perf_counter()-start, device='CPU', dtype='float64'))
    (out / 'results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    return summary  # Notebook直接使用同一计算入口。


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output', default='outputs')
    print(json.dumps(run(parser.parse_args().output), ensure_ascii=False, indent=2))
