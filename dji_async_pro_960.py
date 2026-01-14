import cv2
import torch
import time
import sys
import os
from threading import Thread
from queue import Queue
from ultralytics import YOLO
from datetime import datetime

# --- 1. 配置参数 ---
VIDEO_PATH = "DJI_20251231222628_0001_V.mp4"
MODEL_PATH = "DJI_VisDrone_12n/yolo12n_3070Ti_9602/weights/best.pt"
OUTPUT_PATH = "Deep_Optimized_DJI_960test1.mp4"
BATCH_SIZE = 4  # 3070Ti 8G 显存可以尝试 4-8，越大显存占用越高，速度越快


class DJIProcessor:
    def __init__(self, video_path, model_path):
        self.device = torch.device("cuda:0")
        self.model = YOLO("DJI_VisDrone_12s/yolo12s_3070Ti_960/weights/best.pt").to(self.device)

        # 视频元数据
        cap = cv2.VideoCapture(video_path)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 队列定义 (限制大小防止内存溢出)
        self.raw_queue = Queue(maxsize=128)
        self.result_queue = Queue(maxsize=128)
        self.stopped = False

    def reader(self):
        """线程1: 负责高速读取视频帧"""
        cap = cv2.VideoCapture(VIDEO_PATH)
        while not self.stopped:
            if not self.raw_queue.full():
                ret, frame = cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.raw_queue.put(frame)
            else:
                time.sleep(0.001)
        cap.release()

    def inference(self):
        """线程2: 负责 GPU 推理"""
        # 使用生成器模式配合 stream=True
        results_gen = self.model.predict(
            source=VIDEO_PATH,
            imgsz=960,
            device=self.device,
            stream=True,
            half=True,  # 强烈建议开启：3070Ti 下不损失精度且显著降温、提速
            conf=0.15,  # 权衡值：0.15 可能会导致画面背景“闪烁”虚警，0.2 更稳
            iou=0.7,  # 保持 0.7：密集场景必须放宽 IOU，防止并排的人被剔除
            agnostic_nms=False,  # 关键：设为 False。如果行人和自行车重叠，两者都会保留
            max_det=4000,  # 必须调大：VisDrone 4K 场景一帧可能有几百个目标，默认 300 可能不够
            augment=False,  # 实时推理建议 False，如果追求极致精度且不计成本可设为 True
            classes=[0, 1, 2],  # 如果你只关心车和人，可以指定类别索引，如 [0, 1, 2]
            verbose=False
        )

        for result in results_gen:
            if self.stopped: break
            # 将 GPU 结果放入结果队列
            self.result_queue.put(result)
        self.stopped = True

    def writer(self):
        """线程3: 负责绘制 UI 并写入硬盘"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, self.fps, (self.width, self.height))

        processed_count = 0
        start_time = time.time()

        while not (self.stopped and self.result_queue.empty()):
            if not self.result_queue.empty():
                result = self.result_queue.get()

                # 绘制目标框
                annotated_frame = result.plot(line_width=2)

                # 提取统计数据
                if result.boxes is not None:
                    counts = result.boxes.cls.int().unique(return_counts=True)
                    # 简单绘制总数，减少 CPU 绘图负担
                    total = len(result.boxes)
                    cv2.putText(annotated_frame, f"Detections: {total}", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                out.write(annotated_frame)
                processed_count += 1

                # 性能实时看板
                if processed_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = processed_count / elapsed
                    vram = torch.cuda.memory_reserved() / 1e9
                    sys.stdout.write(f"\r🔥 异步引擎全力运行: {processed_count}/{self.total_frames} | "
                                     f"速度: {fps:.1f} FPS | 显存: {vram:.2f}GB")
                    sys.stdout.flush()
            else:
                time.sleep(0.001)

        out.release()

    def run(self):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 开启生产者-消费者高性能模式...")

        # 启动三个独立线程
        t_read = Thread(target=self.reader)
        t_infer = Thread(target=self.inference)
        t_write = Thread(target=self.writer)

        t_read.start()
        t_infer.start()
        t_write.start()

        t_read.join()
        t_infer.join()
        t_write.join()
        print(f"\n✅ 处理完成！输出视频: {OUTPUT_PATH}")


if __name__ == "__main__":
    processor = DJIProcessor(VIDEO_PATH, MODEL_PATH)
    processor.run()