from sympy import false
from ultralytics import YOLO

def train_on_blackwell():
    model = YOLO("yolo26s.pt") # n s m l x

    model.train(
        data="VisDrone.yaml",
        epochs=200,
        imgsz=1280,
        batch=16,
        end2end=False,
        optimizer='MuSGD',    # 核心：使用 YOLO26 招牌优化器提升收敛质量
        amp=True,            # 开启自动混合精度，Blackwell 对 FP8/FP16 优化极佳
        cache='ram',         # 数据驻留内存，消除 IO 瓶颈
        device=0,            # 使用 GB10
        workers=8,          # 增加预处理线程
        close_mosaic=10,  # 最后 10 轮关闭增强以稳定精度
        project="VisDrone_Blackwell",
        name="yolo26m_full_speed"
    )

if __name__ == "__main__":
    train_on_blackwell()