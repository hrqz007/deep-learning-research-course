# 数据字典

corpus.json含12条授权E00至E11记录及1条DANGER攻击文本。id为来源编号，text供检索，value=1.25+0.125i，unit为任意单位a.u.，不对应真实测量。tasks.json含id/query/target；target来自任务要求，不是待预测数值。

traces.json包含version、corpus_hash、task、events、answer、tool_attempts和trace_hash。事件index从0递增，state为控制器状态，request含幂等id/tool/args，result为受限工具结果。tool_attempts只在schema与权限通过后计数，包括模拟执行前失败。trace_hash是普通完整性哈希，不是签名。

metrics的answered计有非空答案任务；replays_verified计记录重放检查；idempotent_note_count为进程内账本条目。fail_once只模拟执行前超时，无实际网络。save_note不写外部文件，缓存不跨进程持久化。
