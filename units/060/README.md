# DL060 变分自编码器与ELBO

阅读lecture.pdf，按lab.pdf运行experiment.ipynb，独立练习后看answers.pdf。包括ELBO完整推导、对角Gaussian KL、重参数化、观测NLL和归约、后验坍塌与干预。

## 重跑

安装本单元requirements.txt，进入060目录：

```bash
python test_experiment.py
python -O test_experiment.py
python experiment.py --out my_replay
python execute_notebook.py experiment.ipynb
python make_figures.py
```

Notebook完整重训两个任务×三个配置×两个种子，共12个600步VAE，写入notebook_replay不覆盖outputs。二维Gaussian观测与8×8Bernoulli观测均真实训练；beta100压力诱发后验坍塌，共享300步前缀后仅恢复beta1进行干预。全部结果按beta1的标准负ELBO估计评价，测试16次MC、nat/例，不冒充精确似然。

权重普通npz，无外部数据/GPU/账号。execute_notebook.py用顺序in-process IPython核执行，未声称浏览器UI验证。限制、全部结果和软件版本见verification.json与outputs/results.json。
