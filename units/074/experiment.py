"""有限假设类一致收敛：精确总体风险与实际重复抽样比较。"""
from pathlib import Path  # 保存材料到指定目录。
import argparse,json,platform,time  # 命令行、JSON、版本与计时。
import numpy as np  # 数组按[假设,样本]组织。
from generate_data import population,sample  # 原创总体与抽样器。


def radius(n, size, delta=.05):
    """双侧Hoeffding+并集界；原始半径可以超过1，不隐瞒宽松性。"""
    if n < 1 or size < 1 or int(n) != n or int(size) != size or not 0 < delta < 1:
        raise ValueError('positive integer n,size and 0<delta<1 required')
    return float(np.sqrt(np.log(2 * size / delta) / (2 * n)))  # 自然对数。


def true_risks(predictions, prob):
    errors = np.where(predictions == 1, 1 - prob[None, :], prob[None, :])  # 条件错率。
    return errors.mean(axis=1)  # 格点等概率，总体精确求和。


def empirical_risks(predictions, index, y):
    return (predictions[:, index] != y[None, :]).mean(axis=1)  # 每个假设平均0/1损失。


def memorize(index, y, size=21):
    """数据依赖表：已见格点多数表决，未见格点默认0；不是单元素预定类。"""
    counts = np.bincount(index, minlength=size)  # 每个输入出现次数。
    ones = np.bincount(index, weights=y, minlength=size)  # 对应正标签数。
    return (ones > counts / 2).astype(int)  # 并列或未见时取0，规则固定。


def run(output='outputs'):
    start=time.perf_counter(); out=Path(output); out.mkdir(parents=True, exist_ok=True)
    pop=population(); pred=pop['predictions']; truth=true_risks(pred,pop['prob'])
    rows=[]; arrays={'true_risks':truth, **pop}; repeats=400  # 每个n报告全部400次。
    for n in [20, 100, 500, 2000]:
        records=[]  # 每次记录一致偏差、ERM经验/总体风险、记忆器风险。
        for repeat in range(repeats):
            index,y=sample(n,740000+n*1000+repeat)  # 每组可复现，组间不复用样本。
            risks=empirical_risks(pred,index,y); chosen=int(np.argmin(risks))  # 并列取最小索引。
            mem=memorize(index,y); mem_emp=float(np.mean(mem[index]!=y))
            mem_true=float(true_risks(mem[None,:],pop['prob'])[0])  # 精确而非测试近似。
            records.append([np.max(np.abs(risks-truth)),risks[chosen],truth[chosen],
                            mem_emp,mem_true,chosen])
        records=np.asarray(records); eps=radius(n,len(pred))  # 预定有限类界。
        arrays[f'n{n}']=records  # 原始重复结果保留，均值不是全部证据。
        rows.append(dict(n=n, repeats=repeats, H=len(pred), epsilon=eps,
                         violation_count=int(np.sum(records[:,0]>eps)),
                         mean_uniform_gap=float(records[:,0].mean()),
                         mean_train=float(records[:,1].mean()),mean_true=float(records[:,2].mean()),
                         max_erm_excess=float((records[:,2]-truth.min()).max()),
                         memory_mean_train=float(records[:,3].mean()),memory_mean_true=float(records[:,4].mean()),
                         wrong_singleton_radius=radius(n,1), full_table_radius=radius(n,2**21)))
    np.savez_compressed(out/'trials.npz',**arrays)  # 全部原始统计数组。
    result=dict(unit='074',catalog_id='T2',delta=.05,H=len(pred),best_true_risk=float(truth.min()),
                population='21 equiprobable inputs; label probabilities 0.15 or 0.85',rows=rows,
                runtime=dict(python=platform.python_version(),numpy=np.__version__,
                             seconds=time.perf_counter()-start,device='CPU',dtype='float64'))
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs')
    print(json.dumps(run(parser.parse_args().output),ensure_ascii=False,indent=2))
