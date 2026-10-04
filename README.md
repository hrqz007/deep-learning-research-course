# 深度学习研究型课程

从零基础走向可复查研究的中文课程。完整大纲包含 **72个主线单元与30个进阶单元，共102个单元**。按先修关系选择路线，编号不总是学习顺序。

## 已完成的完整单元

当前可学习的正文、实验与练习资料为001–015；其余87个单元仍在制作，不把大纲条目当作已完成教材。

|单元|主题|讲义|独立实验|练习详解|代码与运行说明|
|---|---|---|---|---|---|
|001|深度学习问题与证据|[8页](units/001/lecture.pdf)|[3页](units/001/lab.pdf)|[2页](units/001/answers.pdf)|[进入单元](units/001/README.md)|
|002|Anaconda与Python第一份实验|[13页](units/002/lecture.pdf)|[6页](units/002/lab.pdf)|[4页](units/002/answers.pdf)|[进入单元](units/002/README.md)|
|003|函数文件与最小测试|[10页](units/003/lecture.pdf)|[3页](units/003/lab.pdf)|[2页](units/003/answers.pdf)|[进入单元](units/003/README.md)|
|004|数组张量与形状思维|[9页](units/004/lecture.pdf)|[3页](units/004/lab.pdf)|[2页](units/004/answers.pdf)|[进入单元](units/004/README.md)|
|005|数据管线与可视化检查|[10页](units/005/lecture.pdf)|[4页](units/005/lab.pdf)|[3页](units/005/answers.pdf)|[进入单元](units/005/README.md)|
|006|第一次可重跑的研究练习|[8页](units/006/lecture.pdf)|[3页](units/006/lab.pdf)|[2页](units/006/answers.pdf)|[进入单元](units/006/README.md)|
|007|代数函数与图像语言|[9页](units/007/lecture.pdf)|[5页](units/007/lab.pdf)|[4页](units/007/answers.pdf)|[进入单元](units/007/README.md)|
|008|向量内积与几何|[8页](units/008/lecture.pdf)|[3页](units/008/lab.pdf)|[2页](units/008/answers.pdf)|[进入单元](units/008/README.md)|
|009|矩阵线性映射与仿射变换|[14页](units/009/lecture.pdf)|[4页](units/009/lab.pdf)|[5页](units/009/answers.pdf)|[进入单元](units/009/README.md)|
|010|导数积分与局部变化|[10页](units/010/lecture.pdf)|[3页](units/010/lab.pdf)|[2页](units/010/answers.pdf)|[进入单元](units/010/README.md)|
|011|偏导梯度与链式法则|[12页](units/011/lecture.pdf)|[4页](units/011/lab.pdf)|[5页](units/011/answers.pdf)|[进入单元](units/011/README.md)|
|012|雅可比与矩阵求导|[10页](units/012/lecture.pdf)|[4页](units/012/lab.pdf)|[3页](units/012/answers.pdf)|[进入单元](units/012/README.md)|
|013|秩奇异值与条件数|[12页](units/013/lecture.pdf)|[4页](units/013/lab.pdf)|[4页](units/013/answers.pdf)|[进入单元](units/013/README.md)|
|014|浮点数与稳定数值计算|[8页](units/014/lecture.pdf)|[3页](units/014/lab.pdf)|[3页](units/014/answers.pdf)|[进入单元](units/014/README.md)|
|015|局部近似与优化几何|[12页](units/015/lecture.pdf)|[4页](units/015/lab.pdf)|[4页](units/015/answers.pdf)|[进入单元](units/015/README.md)|

每个单元包含对应Markdown、已执行Notebook、Python脚本、原创合成数据、完整练习答案、来源与核验记录。先读讲义和实验指南，再运行代码；不要只阅读Notebook留下的旧输出。

## 完整路线

- [49页课程大纲](catalog/curriculum.pdf)
- [102单元逐项大纲](catalog/units.md)
- [机器可读先修与路线](catalog/curriculum.json)
- [逐单元制作状态](manifest.json)

新学习者从001开始，002从变量、列表、条件与循环教起，003再建立可测试的函数、文件和最小类接口。暂时不需要GPU、付费API或深度学习框架。

## 运行与验证

001–003、006、007与010核心实验仅用Python标准库；004、005、008、009、011–015使用NumPy，部分Notebook或重建图需要Matplotlib和中文字体。Notebook需要JupyterLab与ipykernel。每单元README和environment.yml给出说明；[共享运行说明](shared/README.md)提供课程环境入口。

这十五个单元已做独立内容与数值核对、脚本新进程执行、Notebook顺序执行、全部PDF页面视觉检查及文件摘要核对。详细范围见[001–003核验说明](releases/001-003.json)、[004–006核验说明](releases/004-006.json)、[007–009核验说明](releases/007-009.json)、[010–012核验说明](releases/010-012.json)与[013–015核验说明](releases/013-015.json)。Notebook使用新进程内的真实IPython InProcessKernel；未声称验证Jupyter浏览器界面、外进程内核传输或所有操作系统上的Anaconda安装。

如需从Markdown重建教材PDF，参见[可选构建工具说明](shared/build-tools/README.md)。阅读现成PDF与运行实验不需要文档构建依赖。

## 资料范围

全部教学校准记录为原创合成数据。公开测试答案适合练习流程，不能当作未见盲测。小实验仅支持正文写明的结论，不证明真实部署效果。参考资料使用一手作者材料与官方文档，图和教学叙事独立编写。

仓库公开供学习。公开可读不等于为第三方参考内容授予额外版权许可；本仓库不替参考文献的作者变更其许可条件。
