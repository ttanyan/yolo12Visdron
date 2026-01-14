from ultralytics import YOLO

def train():

    model = YOLO("yolo12s.pt")  # n s m l x

    # 针对大疆 4K 视频流优化
    model.train(
        data="test.yaml",          # 数据集配置文件路径
        epochs=200,                # 训练轮数，注意力机制需要更多轮次收敛
        # imgsz=1280,                # 输入图像尺寸，3070 Ti 显存8G,若 OOM 则降至 640
        imgsz=960,                 # 输入图像尺寸，3070 Ti 显存8G,若 OOM 则降至 640
        batch=4,                   # 批次大小，3070 Ti (8G) 跑 s 版 960 分辨率建议 batch=12 左右
        optimizer="AdamW",         # 优化器类型，官方建议注意力模型必须使用 AdamW
        lr0=0.001,                 # 初始学习率
        amp=True,                  # 是否开启自动混合精度，加速训练
        nbs=64,                    # 标称批大小，用于梯度累积计算
        workers=4,                 # 数据加载线程数
        project="DJI_VisDrone_12s",    # 项目名称，保存结果的文件夹名
        name="yolo12s_3070Ti_960", # 实验名称，保存权重的子文件夹名
        device=0                   # 指定GPU设备，使用你的 RTX 3070 Ti
    )

if __name__ == "__main__":
    train()



