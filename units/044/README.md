# 044 深层卷积与残差网络

先修035、036、043。按lecture.pdf、lab.pdf、experiment.ipynb顺序学习，answers.pdf用于独立练习后的核对。三份文本同时保留Markdown源。

内容包括：六参数残差网络逐样本前后向与两次同步更新；恒等/投影块的独立NumPy前后向及全元素有限差分；真实Plain/Residual CNN同宽同深配对实验；完整24次训练、BN尖峰诊断、12个冻结选择后的测试结果与全部错误索引。11张原创图由代码和真实数据生成。

在课程根目录：

```bash
python units/044/test_experiment.py
python -O units/044/test_experiment.py
python units/044/experiment.py --output reproduced-044
python units/044/make_figures.py --results reproduced-044/results.json --output reproduced-044/figures
```

核心观察：同深度残差的平均固定测试准确率略高，但浅层验证交叉熵更差；深层普通网也成功拟合。大步长下两图均有严重推理态尖峰，保持权重仅重估BN可显著恢复。本实验支持具体机制审查，不支持残差无条件获胜或梯度永不消失的说法。

所有数据为原创合成；无模型/数据下载、GPU或API要求。environment.yml与requirements.txt描述学习环境，outputs/environment.json记录实际环境。独立解压后脚本仍可定位data；build_tools包含PDF构建脚本和依赖说明。全部候选与状态见outputs/results.json，作者执行记录见verification.json；该记录不等于独立质量审查结论。

本单元属于完整102讲课程，未修改前后课程大纲。研究扩展与不支持范围详见lab.md和lecture.md。

执行范围说明：发布版Notebook在新的Python进程内创建真实ipykernel InProcessKernel并按顺序执行全部单元格，保存计算输出和PNG。构建环境禁止TCP/IPC socket，未测试外进程socket传输或浏览器Jupyter界面；这不等于已验证任意用户电脑上的Jupyter安装。build_tools/execute_notebook_inprocess.py提供同样的无socket执行路线。
