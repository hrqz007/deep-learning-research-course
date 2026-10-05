# 完整固定实验结果

- results.json：协议、完整字节/BPE词表与合并历史、Unicode探针、精确Fraction手算账本、六模型各split逐文档NLL、unigram基线和实际软件版本。
- training_traces.json.gz：六运行，每个states有121个状态，step0为初始化，step1..120是同步更新后的参数；每个状态有完整E、W、b与对应train_nll_per_target。总720次更新、726个参数状态。
- final_states.json.gz：六套终点E/W/b，独立复算全部测试文档，不需要重新训练。

使用gzip.decompress后json.loads读取；结果JSON内同时记录每个压缩文件的compressed_sha256、raw_sha256、raw_bytes。压缩使用mtime=0消除时间戳差异。文件可以较大，因为不是只存loss曲线。

纯字节/BPE的V为260/276，D均为8，E形状V×8，W为8×V，b为V。输入固定PAD行0不学习，输出头仍包含全部V类。没有删除任一种子；神经模型未超过相应unigram基线的结果保留。

输出关联SHA用于发现多文件写入中断或文件混配。训练/数据校验失败先于任何写入；单文件使用原子替换，但整目录不是事务。
