# 049 词元嵌入与序列数据

正式先修017、024、030。原始UTF-8→训练集字节BPE→整数ID→嵌入查表→mask与下一token目标→共享梯度累加→同步更新。含两样本精确手算的两次完整SGD，以及六个CPU float64低秩bigram固定实验。

- lecture.md / lecture.pdf：完整理论、数值链、11幅原始图、固定实验与限制
- lab.md / lab.pdf：从零操作、独立练习与验收标准
- answers.md / answers.pdf：详细解答与失败诊断
- experiment.ipynb：真实执行Notebook，Run All完整重训六模型
- experiment.py、test_experiment.py、generate_data.py、make_figures.py、create_notebook.py：可运行源码
- data/：原始合成双语文本与SHA；outputs/：全部结果、726完整参数状态、六组最终权重
- environment.yml、runtime_versions.json、sources.md、source-checks.json：环境和一手来源

从课程根运行：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python units/049/test_experiment.py
python -O units/049/test_experiment.py
python units/049/experiment.py --output tmp/049-reproduction
python units/049/make_figures.py --output tmp/049-figures
```

不要删除已保留outputs后再运行测试，测试会独立复算其中全部最终模型。Notebook可从根或单元目录启动；create_notebook.py会重建未执行版本，只有明确需要重建时才调用。

字节BPE是本课明确定义的教学实现，不是SentencePiece、WordPiece或GPT兼容tokenizer。未见合法Unicode可以编码，不等于被理解；随机ID可能不是合法UTF-8。神经模型在本次固定120步预算均未超过unigram，原样保留。token困惑度单位依赖tokenizer；bits/byte只计算规范编码路径，不含模型侧信息，也不是全部路径的文字概率。

数值复现目标为所列CPU环境；未做速度、显存或自然语言泛化声明。离线新核是真实IPython核，浏览器UI与socket未测试。公开候选的作者执行/视觉检查不替代独立科学审核。

## 首次安装与启动

已经能运行030实验的读者可直接使用其独立环境，并先核对版本。新建环境的完整入口如下（不代表已测试所有系统的全新安装）：

```bash
conda env create -f units/049/environment.yml
conda activate dl049
python -c "import torch,numpy; print(torch.__version__,numpy.__version__)"
jupyter lab units/049/experiment.ipynb
```

若使用Python venv而非Conda，可从官方PyTorch CPU索引安装torch==2.7.1，再安装environment.yml列出的其他PyPI包；不要混用不明来源的轮子。安装完成后始终用该环境的python执行测试。Notebook浏览器使用Restart Kernel→Run All；离线核验命令为：

```bash
cd units/049
python ../../shared/execute_notebook_inprocess.py experiment.ipynb
```

图像还需要Noto Sans CJK字体。Debian/Ubuntu可安装系统官方fonts-noto-cjk包，或从Google Noto官方发行安装后，将DL_CJK_FONT设置为本机实际ttc/otf文件路径。默认读取/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc；缺字体会明确报错。本课程不打包字体二进制。

PDF可选重建依赖和命令见shared/build-tools/README.md。安装这些工具只用于可信课程源码，不能把构建器当成不可信HTML/TeX的安全沙箱。
