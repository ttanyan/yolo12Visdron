from ultralytics import YOLO

def train_ultimate():
    # 确保加载的是你效果最好的 yolo12s 权重
    model = YOLO("yolo26s.pt")

    model.train(
        data="VisDrone.yaml",
        epochs=300,            # 300轮能让 mAP 曲线在 200 轮的基础上进一步突破
        imgsz=1280,            # 必须维持 1280 以保证小目标不丢失
        batch=128,             # 128GB 统一内存足以支撑 128 的 Batch Size
        device=0,              # GB10 核心
        workers=32,            # 充分利用 Cortex-X925 多核性能
        optimizer='SGD',       # 超大 Batch 下建议用 SGD 配合余弦退火
        lr0=0.01,
        cos_lr=True,
        close_mosaic=20,       # 最后 20 轮关掉增强，锁定最终精度
        cache='ram',           # 关键：你有 128GB 内存，将数据集全部缓存进 RAM，训练速度起飞
        rect=False,            # 密集场景不建议开启矩形训练，保持正方形输入更利于小目标
        amp=True               # 开启 FP16/BF16 混合精度
    )

if __name__ == "__main__":
    train_ultimate()