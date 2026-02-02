from ultralytics import YOLO

def train_on_blackwell_pro():
    model = YOLO("yolo26m.pt")

    model.train(
        data="VisDrone.yaml",
        epochs=200,
        imgsz=1280,
        batch=32,             # 深度优化 1：拉满显存利用率
        optimizer='AdamW',    # 深度优化 2：牺牲极小精度换取巨大速度提升
        end2end=False,        # 深度优化 3：训练期关闭双标签分配
        amp=True,
        cache='ram',
        device=0,
        workers=24,           # 深度优化 4：加速 CPU 预处理
        warmup_epochs=3,
        close_mosaic=10,      # 最后 10 轮关闭增强以稳定精度
        project="VisDrone_Blackwell_Pro",
        name="speed_optimized"
    )

if __name__ == "__main__":
    train_on_blackwell_pro()