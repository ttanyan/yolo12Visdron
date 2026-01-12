# YOLO12 VisDrone 项目

本仓库包含使用YOLO12在VisDrone数据集上进行无人机/目标检测的代码和训练模型。

## 项目结构

- `dji_async_pro_1280.py`, `dji_async_pro_960.py`: DJI无人机应用的主要推理脚本
- `dji_turbo_multiprocessing.py`: 用于优化性能的多进程实现
- `train_visdrone_yolo12.py`: 在VisDrone数据集上训练YOLO12的脚本
- `test.yaml`: 测试配置文件
- `yolo11n.pt`, `yolo12n.pt`: 预训练模型权重
- `DJI_VisDrone/`: 包含训练结果、权重和评估指标

## 主要特性

- 针对基于无人机的目标检测进行了优化
- 包含不同分辨率的多个训练模型（1280, 960）
- 评估指标和可视化工具
- 支持多进程实时推理

## 使用方法

运行推理脚本来对视频流或图像序列执行检测：

```bash
python dji_async_pro_1280.py
```

或

```bash
python dji_async_pro_960.py
```

## 内容说明

`DJI_VisDrone/` 目录包含：

- 训练好的模型权重
- 训练日志和指标
- 评估结果（混淆矩阵、精确率/召回率曲线）
- 示例输出图像