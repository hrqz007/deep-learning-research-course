# DL086 实验：可追踪的工具沙盒

## 任务边界

构建一个只读实验记录的确定性控制器，并用进程内记账工具演示幂等。外部文档没有授权权力；攻击文本不应创建delete_all工具。这里没有真实LLM、真实网络或文件删除能力，所有故障都是可控模拟。

## 1 运行环境

核心实验和测试仅使用Python标准库。若需要图、PDF与真实内核Notebook，安装requirements.txt。

```bash
python experiment.py
python -m unittest -v test_experiment
python -O -m unittest -v test_experiment
```

生成data/corpus.json、data/tasks.json、outputs/traces.json与metrics.json。不要手动修改生成后的金标准来提高分数。图由make_figures.py根据真实输出重建。

## 2 检索手算

打开corpus，找到E03。查询“experiment E03 measured yield”与它有四个共享token；普通其他记录少一个唯一编号。调用retrieve检查排序。再只查“measured yield”，观察并列按编号排序，解释为什么检索器无法知道用户想要哪条。

检索仅用集合交集，不是语义模型。请添加一个词序不同但token相同的查询，验证结果相同。若想支持中文分词或同义词，需要新的设计与测试，不可把当前英数字正则解析器称为通用检索器。

## 3 schema和权限分别测试

构造正确read_record请求，确认返回id、value、unit。接着尝试加入额外shell字段，预期ValueError；保持字段正确但请求DANGER，预期PermissionError；工具名改成delete_all，仍应PermissionError。三个故障说明格式、资源能力和工具白名单是不同检查。

不要为了使测试“通过”增加真正删除功能。拒绝路径本身就是实验预期。工具允许的内容写在代码中，模型或文档文本无法覆盖。

## 4 跟踪正常与失败状态

运行run_task读取E03，事件应包括RECEIVED、RETRIEVED、VALIDATED、CALLING、OBSERVED、FINISHED。启用fail_once后，第一次CALLING进入RETRYABLE_ERROR，第二次成功；tool_attempts为2。

把max_attempts改为1，预期EXHAUSTED且answer为None。错误不是凭空消失，而是停止并留下证据。尝试缺失E99，预期ABSTAINED，不应调用工具猜值。

## 5 攻击文本检查

阅读DANGER文档，只把它当数据。调用任务target=DANGER，可能进入检索候选，但执行层拒绝读取。单独发送delete_all请求也被白名单拒绝。请在报告写明：这测试控制层权限，不测试真实语言模型是否会被诱导生成危险请求。

一个加强练习是在授权记录的text中放入类似指令，但保持结构化值不变；控制器仍应读取结构化字段。该设计简化了信息来源冲突，不能替代真实文献可信度判断。

## 6 幂等与冲突编号

创建Sandbox，对同一save_note请求调用两次。预期返回相同结果、ledger长度1、calls=1。然后复用同一个id却把text改成另一内容，预期ValueError。每次重试随机生成新id会破坏这个保障。

创建全新的Sandbox后，旧缓存消失。把这一点写成明确限制：本课只做进程内幂等，不提供崩溃恢复或持久exactly-once。不要把内存字典演示包装成生产级事务系统。

## 7 重放与篡改

对run_task返回的bundle调用replay，预期verified=true。复制bundle并修改answer.value，预期哈希检查失败。修改语料数值再重放，预期版本/语料不匹配。重放不会再次执行工具，所以不会产生第二个note。

进一步思考：如果修改者重算trace_hash会怎样？普通哈希不能认证身份；本函数还做部分结构检查，但不是完整状态转换证明。报告应区分意外损坏检测与对抗篡改认证。

## 8 实际验收结果与扩展

原始14任务应有12成功读取、1缺证据停止、1权限拒绝；正常读取共24次模拟工具尝试；14条记录重放通过；幂等note只保存一次。14项测试应在普通与优化模式均通过。

扩展时先增加“执行已提交但回执丢失”模拟，再设计事务化幂等记录。若接入真实模型，保持相同schema与执行权限，保存版本与提示，建立额外注入、单位、超时和终止测试。新工具权限不由课程代码自动授权。
