import cv2
import torch
import time
import sys
import os
from threading import Thread
from queue import Queue, Empty
from ultralytics import YOLO
from datetime import datetime

# --- 1. 配置参数 ---
# 建议确保路径指向你表现最好的 9604 文件夹权重
VIDEO_PATH = "3.mp4"
MODEL_PATH = "DJI_VisDrone/yolo12n_3070Ti_1280/weights/1280best.pt"
OUTPUT_PATH = "Deep_Optimized_DJI_1280test01.mp4"


class DJIProcessor:
    def __init__(self, video_path, model_path):
        # 硬件加速配置
        self.device = torch.device("cuda:0")
        # 加载模型并初始化
        self.model = YOLO(model_path).to(self.device)

        # 视频元数据提取
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 队列定义：针对 16G 内存限制，设为 30 帧安全阈值，防止 OOM
        self.result_queue = Queue(maxsize=30)
        self.stopped = False

    def inference_engine(self):
        """核心推理引擎：负责 GPU 高速运算"""
        # half=True 开启 FP16，在 3070Ti 上可显著降低显存占用并翻倍速度
        results_gen = self.model.predict(
            source=VIDEO_PATH,
            imgsz=1280,
            device=self.device,
            stream=True,
            half=True,  # 强烈建议开启：3070Ti 下不损失精度且显著降温、提速
            conf=0.15,  # 权衡值：0.15 可能会导致画面背景“闪烁”虚警，0.2 更稳
            iou=0.9,  # 保持 0.7：密集场景必须放宽 IOU，防止并排的人被剔除
            agnostic_nms=False,  # 关键：设为 False。如果行人和自行车重叠，两者都会保留
            max_det=2000,  # 必须调大：VisDrone 4K 场景一帧可能有几百个目标，默认 300 可能不够
            augment=False,  # 实时推理建议 False，如果追求极致精度且不计成本可设为 True
            classes=None,  # 如果你只关心车和人，可以指定类别索引，如 [0, 1, 2]
            verbose=False
        )

        for result in results_gen:
            if self.stopped:
                break
            # 阻塞式入队，如果绘图太慢，GPU 会等待 CPU
            self.result_queue.put(result)

        self.stopped = True

    def video_writer_engine(self):
        """写入引擎：负责 CPU 绘图与视频编码保存"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, self.fps, (self.width, self.height))

        processed_count = 0
        start_time = time.time()

        while not (self.stopped and self.result_queue.empty()):
            try:
                # 设置超时防止在队列末尾死锁
                result = self.result_queue.get(timeout=2)

                # 绘制目标框：1280px 下线宽设为 1，避免遮挡微小目标
                annotated_frame = result.plot(
                    line_width=1,
                    labels=True,
                    conf=True
                )

                # 叠加实时检测统计
                total_objects = len(result.boxes)
                cv2.putText(annotated_frame, f"Detections: {total_objects}", (40, 70),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)

                out.write(annotated_frame)
                processed_count += 1

                # 性能实时面板
                if processed_count % 15 == 0:
                    elapsed = time.time() - start_time
                    current_fps = processed_count / elapsed
                    vram_used = torch.cuda.memory_reserved() / 1e9
                    sys.stdout.write(f"\r🚀 1280px 推理中: {processed_count}/{self.total_frames} | "
                                     f"速度: {current_fps:.1f} FPS | 显存: {vram_used:.2f}GB")
                    sys.stdout.flush()

            except Empty:
                continue

        out.release()
        print(f"\n\n✨ 处理完成！")
        print(f"📊 平均处理速度: {processed_count / (time.time() - start_time):.2f} FPS")
        print(f"📁 结果路径: {os.path.abspath(OUTPUT_PATH)}")

    def run(self):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 启动 1280px 高性能推理模式...")

        # 仅开启双线程：GPU 推理线程 + CPU 写入线程
        # predict(stream=True) 已包含高效取帧逻辑，不再需要单独的 reader 线程
        t_infer = Thread(target=self.inference_engine)
        t_write = Thread(target=self.video_writer_engine)

        t_infer.start()
        t_write.start()

        t_infer.join()
        t_write.join()


if __name__ == "__main__":
    # 强制清理一次显存碎片
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    processor = DJIProcessor(VIDEO_PATH, MODEL_PATH)
    processor.run()