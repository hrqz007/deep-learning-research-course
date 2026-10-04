# 原创数值配置

config.json全部数值由教学设计生成，无真实数据或个人信息，均无量纲。

- logit_cases：正上溢、全下溢和尾项下溢三组对照
- rounded_logits：两项相差1的大分数，用来分离输入转换误差
- cancellation_inputs：有理化对照的四个正小数
- difference_x：在各dtype转换后的实际点检查exp导数
- difference_steps：13个请求步长；两dtype共26行，包含故意不可分辨的步长

报告中的参考基于实际存储的输入，生产脚本使用90位Decimal，独立测试用110位或150位Decimal。不是无限精度承诺。朴素公式的非有限输出序列化为null并有独立状态；差分失败数值字段为空，不能当0。程序不改变配置。
