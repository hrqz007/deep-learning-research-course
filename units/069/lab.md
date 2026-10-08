# 第069讲 实验指导书

这次实验的主指标是“同一个中断点能否恢复同一条训练轨迹”。不要先追求最高测试分数。你将保存数据、配置、权重、优化器与随机状态，再检验完整性与恢复语义。

## 一 环境和独立入口

在本单元目录运行。使用Python3.12及requirements.txt中的版本可以接近本次执行环境；实验仅使用CPU且不下载数据。安装软件依赖之后，实际训练和检查离线完成。

```bash
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
python experiment.py --output my_run
```

unittest会在临时目录完整训练并检查，普通模式与-O模式都有效。-O会去掉Python语言的assert，但不会关闭unittest的显式断言。程序默认outputs可覆盖，研究扩展请使用新的输出目录。

## 二 确认数据与拆分

data.npz包含x、y、train_indices和test_indices。读取形状应为384×6和384；训练索引0至255，测试256至383，不相交。标签从公开公式生成。记录这是独立生成的合成样本，不代表任何真实数据来源。

duplicate_demo应报告388条记录、384条唯一特征、4条重复。然后思考：若把float32转换为float64，row.tobytes()键会怎样？这项扩展说明精确字节键并非通用语义去重。

## 三 检查中断点保存内容

用weights_only=True读取resume_epoch12.pt。应包含model、optimizer、torch_rng、order_rng、epoch、config。epoch为12，优化器状态不为空。torch_rng负责Dropout等全局随机消费；order_rng负责训练样本排列。

不要先打印全部权重。更有信息量的是检查键、dtype、形状、轮数和是否存在优化器计数。讲义中的snapshot使用deepcopy，避免随后训练修改已保存的内存快照。可在小模型上手动验证：取浅state_dict后更新参数，再看旧字典是否变化。

## 四 比较三条真实训练路径

continuous_epoch24.pt是未中断路径；resumed_epoch24.pt完整恢复第12轮状态后继续；weights_only_epoch24.pt只加载参数后重新开始优化器与随机过程。它们都执行剩余12轮，但这不保证轨迹一样。

固定结果中，前两条最大参数差与训练损失差都为0，准确率0.9453125。权重重启最大参数差约0.294915，准确率0.9296875。你应逐键比较张量，而不只是比较四舍五入后的准确率。同一准确率也可能对应不同参数与概率。

如果完整恢复不相等，优先检查恢复顺序、Dropout状态、优化器、数据顺序与依赖版本。不要先把容差改大掩盖问题。跨平台时可以另设数值容差，但应把新标准与原来的逐位验证区分。

## 五 验证资产清单

asset_manifest.json覆盖6个核心资产：data.npz、config.json和4个检查点。逐个计算SHA256，应无失败。清单不包含自己和results.json，它不是对整个目录全部文件的承诺。

在复制出的目录改变config.json的一处字符，再运行verify_assets；它应指出config.json。不要直接损坏课程固定outputs。verify_assets也会拒绝跳出指定根目录的相对路径，这防止清单中的路径被误当作任意文件读取指令。

哈希比对通过只说明与受信清单一致。若清单本身来自未知来源，不能依靠自洽摘要证明文件可信。把这个限制写在报告里，而不是只贴“PASS”。

## 六 checkpoint重算与Notebook

```bash
python make_figures.py
python create_notebook.py
python execute_notebook.py experiment.ipynb
```

Notebook实际重新运行，并在notebook_outputs保存自己的资产。它会重载三份终点检查点，重算128个测试样本的交叉熵与准确率，检查与运行记录一致，并生成嵌入式曲线和参数差图。Notebook显示的值来自执行，不是手填截图。

固定图由outputs/results.json生成。重建PDF需运行本地build_pdf.py读取lecture.md、lab.md和answers.md；公式和图不依赖在线服务。阅读预生成PDF不需要这些构建依赖。

## 七 做一次有边界的故障注入

一次只遗漏一种状态，例如只遗漏optimizer、只遗漏torch_rng或只遗漏order_rng。在不同输出目录保留结果；其余数据、参数、学习率和剩余轮数保持不变。比较第一步损失、最终参数差与测试指标。

这个扩展比一次同时删掉所有状态更适合定位作用。不过本课已执行的错误分支同时遗漏优化器和两种随机状态，只能证明“不完整恢复有差异”，不能定量分解哪一项贡献最大。研究报告必须保留这个区别。

## 八 最小交付清单

提交数据说明、模型卡、配置、完整运行命令、环境版本、核心资产摘要与恢复对照。说明哪些许可是原创资产授权，哪些来自第三方依赖。结论写成“在某个已测试CPU环境完整恢复一致”，而不是“任何机器任何时间都能完全复现”。
