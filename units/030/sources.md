# 一手来源与实际读取范围

核查日期2026-10-04。核心API按实际执行的PyTorch2.7文档核查；教程站页面页头已较新，教程仅用于保存/加载原则，具体行为另由本机2.7.1+cpu执行验证。正文与数据、图、实验独立撰写，没有复制商业教材。

- S1 [PyTorch2.7 nn.Module](https://docs.pytorch.org/docs/2.7/generated/torch.nn.Module.html)：实际读取子模块注册、super初始化、forward调用入口、parameters/buffers、register_buffer、eval、load_state_dict与state_dict浅拷贝段落。支撑正文第2、6、7节。
- S2 [PyTorch2.7 torch.utils.data](https://docs.pytorch.org/docs/2.7/data.html)：实际读取map-style Dataset、DataLoader、默认批次、generator、drop_last、num_workers和prefetch说明。支撑正文第4、7节；未运行多worker或分布式加载。
- S3 [PyTorch2.7 SGD](https://docs.pytorch.org/docs/2.7/generated/torch.optim.SGD.html)：实际读取更新式、首步动量缓冲、foreach、state_dict与load_state_dict说明。本讲无weight decay/dampening/Nesterov，使用foreach=False；600次Fraction递推和实际框架更新核验。
- S4 [PyTorch2.7 Dropout](https://docs.pytorch.org/docs/2.7/generated/torch.nn.Dropout.html)：实际读取训练时独立屏蔽、1/(1-p)缩放、eval身份映射与shape说明。本讲未评估Dropout的普遍正则化收益。
- S5 [PyTorch Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)：实际读取state_dict、最佳状态深拷贝、general checkpoint、优化器状态、weights_only与模式恢复段落；页面教程版本较新，采用的调用在实际2.7.1运行。
- S6 [PyTorch2.7 Reproducibility](https://docs.pytorch.org/docs/2.7/notes/randomness.html)：实际读取平台/版本/CPU与GPU复现限制、torch.manual_seed、其他随机源与确定性算法说明。本实验只验证记录环境，不把固定种子等同于跨环境逐位复现。
- S7 [PyTorch2.7 torch.load](https://docs.pytorch.org/docs/2.7/generated/torch.load.html)：实际读取map_location、weights_only的受限对象类型、反序列化可信来源警告。只加载本课程自生成的检查点。

补充：ModuleList、Parameter与MSELoss的独立官方页面此次访问失败，没有宣称网页已读。读取实际安装的torch.nn.ModuleList、torch.nn.Parameter、torch.nn.MSELoss、torch.nn.ParameterList及Module.zero_grad文档字符串，结合漏注册探针、shape保护和显式MSE推导进行核验。ParameterList的安装版文档也已读取，但主实验不使用该容器。

合成数据规律、数值结果、状态删减对照和归纳论证属于本课程的原始设计与推导；官方文档不为这些具体实验数值背书。
