"""真实tanh网络与初始点线性化网络同步训练；宽度×幅度×种子全报告。"""
from pathlib import Path  # 指定输出位置。
import argparse,json,platform,time  # 命令行、结果序列化、运行信息。
import numpy as np  # 手写前向、Jacobian与梯度；不调用框架。
from generate_data import generate  # 原创函数数据。


def initialize(width, seed):
    rng=np.random.default_rng(seed)  # 每个配置固定可复现初始化。
    return np.r_[rng.normal(size=width),rng.normal(size=width),rng.normal(scale=.3,size=width)]
    # 参数顺序[a_1..a_m, w_1..w_m, b_1..b_m]，共3m个。


def forward_jacobian(theta,x):
    """返回f和J；J[i,j]是第i个输出对第j个参数的导数。"""
    m=len(theta)//3; a,w,b=np.split(theta,3)  # 还原三组长度m参数。
    h=np.tanh(x[:,None]*w[None,:]+b[None,:])  # 隐层[n,m]。
    scale=np.sqrt(m); f=h@a/scale  # f=sum(a*h)/sqrt(m)，明确参数化。
    slope=(1-h*h)*a[None,:]/scale  # tanh导数乘输出权重。
    jac=np.concatenate([h/scale,slope*x[:,None],slope],axis=1)  # 三组偏导。
    return f,jac,h  # 保留隐层，用于量化表示漂移。


def train(width, amplitude, seed, data, steps=500):
    theta0=initialize(width,seed); theta=theta0.copy()  # 同一初值用于两种训练。
    f0,j0,h0=forward_jacobian(theta0,data['x'])  # 冻结初始Jacobian。
    fg0,jg0,_=forward_jacobian(theta0,data['grid'])  # 网格上的相应初值。
    y=amplitude*data['y']; yg=amplitude*data['grid_y']  # 幅度改变目标距离。
    gram=j0@j0.T; eig=np.linalg.eigvalsh(gram/len(y))  # 对平均平方损失的曲率。
    eta=.3/eig[-1]  # 小于固定核稳定阈值2/lambda_max；非线性无全程保证。
    delta=np.zeros_like(theta0)  # 线性化模型参数增量独立更新。
    records=[]  # 每10步记录损失、预测差、核漂移、表示漂移、参数移动。
    for t in range(steps+1):
        f,j,h=forward_jacobian(theta,data['x']); fl=f0+j0@delta  # 两个模型前向。
        if t%10==0 or t==steps:
            fg,_,_=forward_jacobian(theta,data['grid']); fgl=fg0+jg0@delta
            records.append([t,np.mean((f-y)**2)/2,np.mean((fl-y)**2)/2,
                            np.sqrt(np.mean((fg-fgl)**2))/amplitude,
                            np.linalg.norm(j@j.T-gram)/np.linalg.norm(gram),
                            np.linalg.norm(h-h0)/np.linalg.norm(h0),
                            np.linalg.norm(theta-theta0)/np.linalg.norm(theta0)])
        if t<steps:
            theta-=eta*(j.T@(f-y))/len(y)  # 真实网络每步重算Jacobian。
            delta-=eta*(j0.T@(fl-y))/len(y)  # 线性化网络始终用J0。
    final_grid,final_j,_=forward_jacobian(theta,data['grid'])
    linear_grid=fg0+jg0@delta  # 固定核对应的最终网格预测。
    arrays=dict(theta0=theta0,theta=theta,linear_delta=delta,history=np.asarray(records),
                nonlinear_grid=final_grid,linear_grid=linear_grid,initial_gram=gram)
    row=dict(width=width,amplitude=amplitude,seed=seed,steps=steps,eta=float(eta),
             train_loss=float(records[-1][1]),linear_train_loss=float(records[-1][2]),
             relative_prediction_rmse=float(records[-1][3]),kernel_drift=float(records[-1][4]),
             feature_drift=float(records[-1][5]),relative_parameter_movement=float(records[-1][6]),
             nonlinear_grid_rmse=float(np.sqrt(np.mean((final_grid-yg)**2))),
             linear_grid_rmse=float(np.sqrt(np.mean((linear_grid-yg)**2))),
             initial_max_eigenvalue=float(eig[-1]),initial_min_eigenvalue=float(eig[0]))
    return row,arrays


def run(output='outputs'):
    start=time.perf_counter();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    data=generate();rows=[];arrays=dict(data)  # 同一数据用于全部27个配置。
    for width in [8,32,128]:
        for amplitude in [.2,1.,4.]:
            for seed in [11,23,37]:
                row,trace=train(width,amplitude,seed,data)  # 全量真实重训，无选择。
                key=f'w{width}_a{amplitude:g}_s{seed}'  # 唯一配置标识。
                rows.append(dict(key=key,**row))
                arrays.update({key+'_'+name:value for name,value in trace.items()})
    np.savez_compressed(out/'trajectories.npz',**arrays)  # 保存权重以便重算预测。
    result=dict(unit='075',catalog_id='T3',parameterization='sum(a*tanh(w*x+b))/sqrt(m)',
                loss='mean squared error / 2',schedule='500 full-batch GD steps; eta=0.3/lambda_max(K0/n)',
                rows=rows,runtime=dict(python=platform.python_version(),numpy=np.__version__,
                seconds=time.perf_counter()-start,device='CPU',dtype='float64'))
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs')
    print(json.dumps(run(parser.parse_args().output),ensure_ascii=False,indent=2))
