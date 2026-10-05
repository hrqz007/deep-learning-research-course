# 第049讲备份状态与大文件恢复

本单元为完整作者包备份，独立验收尚未完成。原始教学文件保持不变。

原始轨迹文件超出网页单文件入库限制，因此保存在[本仓库Release资产](https://github.com/hrqz007/deep-learning-research-course/releases/tag/dl049-author-backup-20261005)，不在Git文件树内。仅克隆仓库或下载源码ZIP不会取得该文件。

- 原路径：`units/049/outputs/training_traces.json.gz`
- 原文件大小：30,437,808 字节
- SHA256：`8ad2753061c52b838b8810532f21fbd9cb6bdd47e81da292556a43dd4841982d`
- [直接下载原文件](https://github.com/hrqz007/deep-learning-research-course/releases/download/dl049-author-backup-20261005/training_traces.json.gz)

## 恢复方法

在仓库根目录执行以下命令。保留 `.gz` 压缩格式，不要解压或改名。

```sh
mkdir -p units/049/outputs
curl -fL 'https://github.com/hrqz007/deep-learning-research-course/releases/download/dl049-author-backup-20261005/training_traces.json.gz' -o units/049/outputs/training_traces.json.gz
sha256sum units/049/outputs/training_traces.json.gz
```

SHA256应与上面列出的值完全一致。Windows也可从上述链接下载，把原文件放入同一路径，然后用 `Get-FileHash -Algorithm SHA256 units/049/outputs/training_traces.json.gz` 校验。

该Release资产已实际从公开远端重新下载，大小与SHA256均与原文件一致。这里的字节核对不代表本单元通过独立科学验收。
