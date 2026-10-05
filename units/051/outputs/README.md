# 完整可复算结果

results.json含协议、六模型所有split的teacher logits/逐序列NLL/context、自由输出与gold、错误/停止指标、置零上下文诊断。所有种子都保留。

training_traces.json.gz保存六运行各301个完整参数状态，共1806。step0为初始参数，step1..300为更新后状态；前300状态记录原始全局梯度范数和共同clip因子。final_states.json.gz保存全部终点参数。压缩mtime=0；results同时记录压缩与解压原文SHA。

hand_ledger.json另存三次前向、两次SGD、每个时间步门值/局部上游/共享参数贡献，以及Decimal70独立前向。用hand_calculation.py可重建。

训练和序列化先完成，随后逐文件原子替换；整目录不是事务，发现文件混配应按SHA停止解释。audit_records.py用独立NumPy BPTT重算所有1800次更新，并核clip，不只看loss曲线。
