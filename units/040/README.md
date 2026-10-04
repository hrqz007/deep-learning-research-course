# 040 训练诊断与调参实验

先修022、031、034、037、039。以可核对的曲线、样本和配对对照决定下一次实验，保留失败和选择偏差，避免看图直接断定病因。

## 阅读路径

- [正文](lecture.pdf)与[可检索源码](lecture.md)：完整两样本七参数两轮反向及第三次前向、诊断解释、有限网格、选择偏差、条件性测试区间。
- [独立实验指南](lab.pdf)与[源码](lab.md)：协议、文件字段、复跑、故障保护和提交要求。
- [练习详解](answers.pdf)与[源码](answers.md)：10题完整数值和推导。
- [Notebook](experiment.ipynb)：清空内核后顺序执行，会真正重做51次训练并核对6个数值文件。
- [脚本](experiment.py)、[测试](test_experiment.py)、[绘图](make_figures.py)、[数据生成器](generate_data.py)。
- [数据与协议说明](data/README.md)、[一手来源](sources.md)、[验证范围](verification.json)。

## 运行

按environment.yml建立环境后，在本目录执行：

```bash
python test_experiment.py
python experiment.py
python make_figures.py
```

脚本默认按自身位置找数据，可从任意工作目录用绝对脚本路径调用。显式相对路径按当前工作目录解释。Notebook从040目录或课程仓库根启动；第一格核对源文件、fixture、参考数值与图，再导入实验模块。输出先全部计算和序列化，再逐文件原子替换；不声称整个目录构成崩溃事务。

完整实验51次训练，共12206次接受更新。主搜索9配置×3种子×180步；另有21次诊断、2次小批次探针和1次过拟合探针。3次过大步长运行提前停止，其他48次完成各自预算。全部12257个参数状态、曲线、失败候选和末预测保存，13张原创图据实际记录生成。

outputs含6份数值文件。training_traces.json.gz无损保存原JSON，用标准库gzip读取；固定level9、mtime=0、空内部filename。results.json记录解压长度与SHA。固定环境核对压缩字节；压缩器版本改变时也要检查解压内容，不能把封装变化误称数值变化。

## 结果与限制

主搜索选择学习率0.2、L2=0；固定参照为0.05、0.001。最终3模型均值在512个独立合成测试例上的half-MSE约0.048405与0.050931。按测试ID配对的bootstrap只描述固定训练、选择和模型下的测试抽样不确定性，不覆盖重新调参或分布变化。

保留了饱和比例更高却验证更好的反例，以及置乱8例在有限预算内未完全拟合的结果。不能把诊断指标写成通用定理；也没有把诊断中更好的尺度条件事后塞进主获胜配置。

交付实际运行环境：Python3.12.14、PyTorch2.7.1+cpu、NumPy2.3.5、Matplotlib3.10.8，CPU float64。Notebook使用新Python进程的全新IPython InProcessKernel；浏览器Jupyter UI、跨进程socket传输、新建Anaconda、GPU和跨平台逐位复现未验证。图使用Noto Sans CJK字体；成品PDF阅读不依赖本机装字体。

数据全部原创合成，无个人数据，不下载预训练模型，不调用付费API。作者验证与独立验收、真实发布是三个不同阶段，状态以课程发布记录为准。
