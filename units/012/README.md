# 第012单元 雅可比与矩阵求导

先修：[011 偏导梯度与链式法则](../011/README.md)。本讲回答：反向传播为什么计算向量雅可比积，而不显式存整个雅可比？

## 学习顺序

1. 阅读[正文](lecture.pdf)，先完成两层批次网络的纸笔梯度与shape
2. 按[实验指南](lab.pdf)运行[Notebook](experiment.ipynb)或[脚本](experiment.py)
3. 用[练习详解](answers.pdf)核对推导、手算、错误与解释

正文10页、实验4页、详解3页，配有7张原创说明图。正文与实验对应的可编辑来源在lecture.md、lab.md、answers.md。data/network.json为原创无量纲合成数据；来源核对记录见source-checks.json。

## 运行

已有Python 3.12和NumPy时，在本单元目录执行：

```bash
python experiment.py --output outputs
python test_experiment.py
```

需要独立Anaconda环境可参考environment.yml：

```bash
conda env create -f environment.yml
conda activate dl-unit-012
jupyter lab
```

在Notebook选择该环境内核，重新启动后从头运行。脚本写outputs，Notebook写notebook_outputs；每次覆盖对应目录的results.json、difference_scan.csv、environment.json，不改数据配置。要保留旧记录，脚本可指定不同--output目录。仓库保留一份脚本输出和Notebook内嵌的执行结果，Notebook输出目录运行时生成。

核心依赖只有NumPy，CPU即可，不联网、不调用付费API、不下载模型。可选的图重建需matplotlib和Noto Sans CJK字体，运行python make_figures.py；公开构建工具说明在[shared/build-tools](../../shared/build-tools/README.md)。PDF重建须从课程根目录运行python shared/build_pdf_mathjax.py units/012/lecture.md，并对lab.md与answers.md重复操作；图更新后必须重新渲染目视。

## 关键检查

- 损失29.140625；G_W1=[[-7.375,-50.75],[1.375,59.75]]
- G_b1=[[3.375,56.75]]；G_W2=[[-3.34375],[-34.71875]]；G_b2=[[-5.875]]
- G_X=[[-5,2],[-48.375,56.4375]]
- 显式参数雅可比shape(2,9)，参数顺序W1、b1、W2、b2，各矩阵按行展开
- 全部参数方向0.1的JVP约为[[-0.075],[-0.35]]；权重(1,2)的配对结果约-0.775
- 7组测试，100个Fraction矩形网络夹具、31项非法输入/范围拒绝及6段文档代码
- h=1e-5时13坐标最大差分误差小于2e-8；h=1e-17明确拒绝，不写成零误差

实验代码是手写局部JVP/VJP规则，未实现通用自动微分框架。标签shape须严格相等，偏置需显式一行；原始容器内布尔值和布尔ndarray叶子拒绝，数值ndarray此前丢失的类型来源无法恢复。程序拒绝非有限中间结果；未承诺全float64动态范围或下溢无误差。显式雅可比只允许至多100000个元素。

验证使用新进程脚本、真实IPython InProcessKernel顺序执行12个代码格、全部17页PDF逐页目视，以及独立精确有理数和差分核验。未实际新装Anaconda、跨平台环境、浏览器Jupyter界面或跨进程socket内核。图中数组内存是公式值，不是峰值内存测量；小数学实验不提供训练效果或性能保证。作者验证细节在verification.json；课程发布状态由根目录清单管理。
