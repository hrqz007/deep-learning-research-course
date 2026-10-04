# 深度学习研究型课程

从零基础走向可复查研究的中文课程。完整大纲包含 **72个主线单元与30个进阶单元，共102个单元**。按先修关系选择路线，编号不总是学习顺序。

## 已完成的完整单元

当前可学习的正文、实验与练习资料为001–003；其余单元仍在制作，不把大纲条目当作已完成教材。

|单元|主题|讲义|独立实验|练习详解|代码与运行说明|
|---|---|---|---|---|---|
|001|深度学习问题与证据|[8页](units/001/lecture.pdf)|[3页](units/001/lab.pdf)|[2页](units/001/answers.pdf)|[进入单元](units/001/README.md)|
|002|Anaconda与Python第一份实验|[13页](units/002/lecture.pdf)|[6页](units/002/lab.pdf)|[4页](units/002/answers.pdf)|[进入单元](units/002/README.md)|
|003|函数文件与最小测试|[10页](units/003/lecture.pdf)|[3页](units/003/lab.pdf)|[2页](units/003/answers.pdf)|[进入单元](units/003/README.md)|

每个单元包含对应Markdown、已执行Notebook、Python脚本、原创合成数据、完整练习答案、来源与核验记录。先读讲义和实验指南，再运行代码；不要只阅读Notebook留下的旧输出。

## 完整路线

- [49页课程大纲](catalog/curriculum.pdf)
- [102单元逐项大纲](catalog/units.md)
- [机器可读先修与路线](catalog/curriculum.json)
- [逐单元制作状态](manifest.json)

新学习者从001开始，002从变量、列表、条件与循环教起，003再建立可测试的函数、文件和最小类接口。暂时不需要GPU、付费API或深度学习框架。

## 运行与验证

前三单元核心实验仅用Python标准库，Notebook需要JupyterLab与ipykernel。每单元README和environment.yml给出说明；[共享运行说明](shared/README.md)提供课程环境入口。

这三个单元已做独立内容与数值核对、脚本新进程执行、Notebook顺序执行、全部PDF页面视觉检查及文件摘要核对。详细范围见[本批核验说明](releases/001-003.json)。Notebook使用新进程内的真实IPython InProcessKernel；未声称验证Jupyter浏览器界面、外进程内核传输或所有操作系统上的Anaconda安装。

## 资料范围

全部教学校准记录为原创合成数据。公开测试答案适合练习流程，不能当作未见盲测。小实验仅支持正文写明的结论，不证明真实部署效果。参考资料使用一手作者材料与官方文档，图和教学叙事独立编写。

仓库公开供学习。公开可读不等于为第三方参考内容授予额外版权许可；本仓库不替参考文献的作者变更其许可条件。
