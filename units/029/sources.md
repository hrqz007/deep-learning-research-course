# 来源与复现边界

本讲独立撰写解释、手算、实验和图示；同一课程027的数据与配置原样沿用。2026-10-04查阅下列PyTorch官方2.7文档，实际运行PyTorch2.7.1+cpu；历史版本选择为固定复现，不声称是最新版本。

- [PyTorch tensor](https://docs.pytorch.org/docs/2.7/generated/torch.tensor.html)：copy construction; detached new leaf; dtype/device。
- [PyTorch from_numpy](https://docs.pytorch.org/docs/2.7/generated/torch.from_numpy.html)：shared storage and read-only array warning。
- [PyTorch is_leaf](https://docs.pytorch.org/docs/2.7/generated/torch.Tensor.is_leaf.html)：leaf definition; nonleaf retain_grad。
- [PyTorch detach](https://docs.pytorch.org/docs/2.7/generated/torch.Tensor.detach.html)：history cut and storage sharing。
- [PyTorch Tensor.to](https://docs.pytorch.org/docs/2.7/generated/torch.Tensor.to.html)：same object if dtype/device match; conversion copy。
- [PyTorch backward](https://docs.pytorch.org/docs/2.7/generated/torch.Tensor.backward.html)：gradient seed, accumulation, retain_graph/create_graph。
- [PyTorch autograd.grad](https://docs.pytorch.org/docs/2.7/generated/torch.autograd.grad.html)：VJP, returned rather than accumulated gradients; higher-order graph。
- [PyTorch no_grad](https://docs.pytorch.org/docs/2.7/generated/torch.no_grad.html)：new-operation tracking, factory-function exception, backward-mode scope。
- [PyTorch autograd mechanics](https://docs.pytorch.org/docs/2.7/notes/autograd.html)：history and saved tensors; grad modes and eval; inplace version checks。
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.7/generated/torch.nn.CrossEntropyLoss.html)：logits/class-index target, reduction and class-weight denominator。
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html)：floating-point nonassociativity and no cross-platform bitwise guarantees。
- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/#v271)：v2.7.1 CPU wheel installation for Linux/Windows and separate OSX instructions。

Tensor.retain_grad单独网页未成功载入，因此使用已读is_leaf文档及实际安装包的retain_grad、clone文档字符串交叉核查，没有把网页失败当成已读。别名外部写入的危险及具体结果来自本讲隔离CPU探针，不承诺各版本都以同一种方式静默失败。

安装来源为PyTorch官方CPU轮子与其必要依赖，运行代码不联网。官方文档是API语义依据，不能代替本讲手算、NumPy参照、标量差分或语义回归。未测试GPU、浏览器Jupyter、socket传输、新Anaconda安装或跨平台安装。
