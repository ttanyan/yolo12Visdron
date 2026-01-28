from ultralytics import YOLO
import os

# 优化 GB10 的显存管理
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def train_yolo26s_visdrone():
    model = YOLO("yolo26s.pt")

    model.train(
        data="VisDrone.yaml",
        epochs=300,            # 300轮能充分利用注意力机制的学习能力
        imgsz=1280,            # 针对 4K 航拍，1280 是保证小目标精度的底线
        batch=128,             # 128GB 内存完全可以支撑 128 甚至更高的 Batch
        device=0,              # 使用 GB10
        workers=20,            # 匹配你的 20 核 CPU 架构
        optimizer='AdamW',     # 注意力模型推荐使用 AdamW 以获得更好的收敛
        lr0=0.001,             # 学习率起始值
        cos_lr=True,           # 余弦退火策略
        cache='ram',           # 极致加速：将数据全部塞进 128GB 内存
        close_mosaic=20,       # 最后 20 轮关闭增强以稳定精度
        amp=True               # 开启混合精度训练
    )

if __name__ == "__main__":
    train_yolo26s_visdrone()