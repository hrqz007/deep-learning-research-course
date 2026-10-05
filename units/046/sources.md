# 一手来源与适用范围

检索日期：2026-10-05。教学推导、合成数据、图、数值例子和实现均为本单元原创；以下一手资料用于核对结构思想、接口和评估约定，不复用其图片。

1. Faster R-CNN，Ren等，论文§III-A，https://arxiv.org/html/1506.01497v3 。核对参考框编码、训练分配与分类/回归分工。本讲不复现该检测器。
2. Torchvision 0.22 NMS官方文档，https://docs.pytorch.org/vision/0.22/generated/torchvision.ops.nms.html 。核对XYXY输入、IoU严格大于阈值抑制、同分CPU/GPU差异。
3. COCO官方cocoeval.py，https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/cocoeval.py 。核对0.50:0.05:0.95、101召回点、maxDets、crowd/ignore、面积与缺失类别处理。master可能变化，核对日期见上；本讲教学函数不是官方替代。
4. PyTorch 2.7 BCEWithLogitsLoss官方文档，https://docs.pytorch.org/docs/2.7/generated/torch.nn.BCEWithLogitsLoss.html 。核对稳定logit损失、pos_weight与归约。
5. U-Net，Ronneberger等，https://arxiv.org/abs/1505.04597 。核对收缩/扩张与精细定位的结构动机，只作概念延伸。
6. Feature Pyramid Networks，Lin等，https://arxiv.org/abs/1612.03144 。核对多尺度特征融合动机，只作概念延伸。

初次访问旧域pytorch.org的NMS文档失败，随后使用当前官方docs.pytorch.org文档成功；未用二手材料补写失败页面内容。所有具体实验成绩来自本地真实执行，不来自论文或示例输出。
