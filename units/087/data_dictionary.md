# 数据字典

tasks_frozen.json含80题，exact/alias/missing/denied各20。id唯一；query为控制器可见请求；expected只供评分器，含预期status与可回答题的id/value。corpus.json含40条E000至E039，value=2+0.125i，unit=a.u.。所有记录合成。控制器输入显式排除expected/group。

traces.json有5版本×3种子×80题=1200条记录，各记录含result与score。success要求可回答题值及引用正确，否则要求预期停止状态；value_only只不检查引用；retrieval_hit在不可回答题为null；tool_failure指遇过模拟瞬时失败，可恢复。

成本为检索/控制固定1单位加每次mock尝试2单位，不是时间、价格或token。最大2次调用，故障概率0.3且仅首次可能失败。per_task_mean_success先平均3种子，bootstrap以80题成对重采样4000次，区间对应当前经验任务分布。普通bootstrap未固定四组配比。
