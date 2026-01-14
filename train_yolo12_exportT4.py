from ultralytics import YOLO

def export():

    # 在 T4 环境下加载训练好的权重并导出
    model = YOLO("best.pt")
    model.export(format="engine", int8=True, data="test.yaml", device=0)

if __name__ == "__main__":
    export()

# from ultralytics import YOLO
#
# # 加载 YOLO12 模型
# model = YOLO("yolo12s.pt")
#
# # 针对无人机视角训练，建议增大 imgsz
# model.train(
#     data="uav_data.yaml",
#     epochs=100,
#     imgsz=960,       # 提高分辨率以识别小目标
#     device=[0, 1],   # 使用双 T4 显卡
#     batch=16         # 根据显存调整
# )



# from ultralytics import YOLO
#
# # 加载模型
# model = YOLO("yolo12x.pt")
#
# # 在两张 T4 (GPU 0 和 1) 上进行训练
# model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[0, 1])
