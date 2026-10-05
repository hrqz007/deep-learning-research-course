# 051 门控记忆与编码解码

正式先修050。GRU/LSTM门控机制、完整两样本两次SGD数值链、encoder-decoder、teacher forcing与自由生成。主实验两模型各三种子，固定反序列协议，共1800次更新；保留GRU不如RNN和长度外推全部失败的负面结果。

## 文件

- lecture.md / lecture.pdf：完整主线、公式与12张原创图
- lab.md / lab.pdf：独立操作步骤、数值与编程练习
- answers.md / answers.pdf：全部练习详解和结果解释
- experiment.ipynb：实际新核执行；Run All重训全部六模型并复算所有更新
- experiment.py：基础算子RNN/GRU、独立NumPy反传、完整训练和自由解码
- hand_calculation.py：14参数的逐样本逐时间账本，70位Decimal前向
- test_experiment.py、audit_records.py：cell/完整模型参照、FD、所有记录更新核对
- data/、outputs/：原创数据、协议、完整1806参数状态、最终模型和全部生成
- generate_data.py、make_figures.py、create_notebook.py：可再生成源码
- environment.yml、runtime_versions.json、sources.md、source-checks.json：环境与一手来源

## 建立独立环境并运行

从课程根运行。已有050环境可先检查实际版本，无需重复安装；新环境入口：

```bash
conda env create -f units/051/environment.yml
conda activate dl051
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python units/051/test_experiment.py
python -O units/051/test_experiment.py
python units/051/experiment.py --output tmp/051-reproduction
python units/051/audit_records.py --output tmp/051-reproduction
jupyter lab units/051/experiment.ipynb
```

CPU torch可按PyTorch官方安装渠道选择2.7.1 CPU wheel；不要求CUDA或GPU。不要先删除保留outputs，终点重载测试需要它们。自己的实验输出写新目录，不覆盖原件。环境文件不是所有操作系统全新安装或任意硬件位级一致的保证。

## 图和完整Notebook需要中文字体

数值脚本不依赖字体，但make_figures.py和完整Notebook需要Noto Sans CJK。当前验证Linux路径/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc；其他位置通过DL_CJK_FONT指定实际ttc/otf文件。Debian/Ubuntu可从官方系统仓库安装并检查：

```bash
sudo apt-get update
sudo apt-get install fonts-noto-cjk
cd units/051
python -c "import make_figures; print('CJK font ready')"
```

若缺字体，先解决字体或路径，不继续绘图与完整Notebook；已有PDF可直接阅读。其他系统需安装同一字体家族并设置实际路径，未宣称已测试其全新安装。不打包第三方字体二进制。

Notebook可在课程根或单元目录启动。离线验收入口在单元目录执行：

```bash
python ../../shared/execute_notebook_inprocess.py experiment.ipynb
```

这个命令会更新该Notebook，核对冻结产物请先复制单元。它使用新Python进程中的真实IPython核，不代表已测试浏览器UI或外进程socket。create_notebook.py会重建未执行源，只有要重新执行时才调用。

## PDF重建与范围

普通实验不需要MathJax/WeasyPrint。可选PDF重建方法见shared/build-tools/README.md；从课程根用shared/build_pdf_mathjax.py处理本讲三个Markdown。此工具只用于可信课程源码。重建后必须逐页目视。

本课采用PyTorch reset-after GRU，不声称与所有论文变体权重兼容。RNN/GRU同H和步数但参数/计算量不同；LSTM只做解析cell例，不另训练第三架构。数字只属于固定合成任务，teacher token、自由token和严格EOS序列exact分别报告，不能互换。

作者执行/视觉检查不等于独立QA PASS，最终冻结与独立验收由课程流程另行记录。
