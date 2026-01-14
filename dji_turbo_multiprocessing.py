import cv2
import torch
import numpy as np
import subprocess
from ultralytics import YOLO
from multiprocessing import Process, Queue, Manager
import time

# --- 1. 性能参数配置 ---
BATCH_SIZE = 4  # 每次推理 4 帧，充分利用 3070Ti 算力
IMG_SIZE = 960  # 训练时的分辨率
VIDEO_PATH = "DJI_20251231200946_0001_V.mp4"
OUTPUT_PATH = "Turbo_NVENC_Output.mp4"


def frame_reader(video_path, task_queue):
    """【进程 A】读取视频：全力读取原始帧并压入队列"""
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        # 如果队列太满则稍等，防止撑爆内存
        while task_queue.qsize() > 100: time.sleep(0.01)
        task_queue.put(frame)
    cap.release()
    task_queue.put(None)  # 结束标志


def gpu_inference(task_queue, result_queue, model_path):
    """【进程 B】GPU 推理：负责最核心的 AI 计算"""
    device = torch.device("cuda:0")
    model = YOLO(model_path).to(device)

    # 启用 FP16 半精度和性能优化
    model.model.half()

    batch = []
    while True:
        frame = task_queue.get()
        if frame is None: break

        batch.append(frame)

        # 当凑够一个 Batch 或者收到结束信号时进行推理
        if len(batch) == BATCH_SIZE:
            # 批量推理提升 GPU 利用率
            results = model.predict(batch, imgsz=IMG_SIZE, device=device, half=True, verbose=False)
            for res in results:
                result_queue.put(res)
            batch = []

    result_queue.put(None)


def video_writer_nvenc(result_queue, width, height, fps):
    """【进程 C】硬编码写入：利用 3070Ti 的 NVENC 芯片，不占 CPU"""
    # 使用 FFmpeg 调用 NVENC 硬件加速硬编码
    command = [
        'ffmpeg',
        '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-',  # 从管道输入
        '-c:v', 'h264_nvenc',  # 关键：调用 NVIDIA 硬件编码器
        '-preset', 'fast', '-b:v', '20M',  # 高码率保证 4K 画质
        OUTPUT_PATH
    ]

    # 开启子进程
    pipe = subprocess.Popen(command, stdin=subprocess.PIPE)

    processed_count = 0
    start_time = time.time()

    while True:
        result = result_queue.get()
        if result is None: break

        # 渲染 (plot 在 CPU 上运行，i9 的多核优势在这里体现)
        annotated_frame = result.plot(line_width=2)

        # 写入 FFmpeg 管道
        pipe.stdin.write(annotated_frame.tobytes())
        processed_count += 1

        if processed_count % 50 == 0:
            avg_fps = processed_count / (time.time() - start_time)
            vram = torch.cuda.memory_reserved() / 1e9
            print(f"\r🔥 硬件全开模式: {processed_count} 帧 | 速度: {avg_fps:.1f} FPS | 显存: {vram:.2f}GB", end="")

    pipe.stdin.close()
    pipe.wait()


if __name__ == "__main__":
    # 获取视频参数
    cap = cv2.VideoCapture(VIDEO_PATH)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # 跨进程通信队列
    task_q = Queue(maxsize=128)
    result_q = Queue(maxsize=128)

    MODEL_WT = "DJI_VisDrone_12n/yolo12n_3070Ti_1280/weights/960best.pt"

    # 启动多进程架构
    p_read = Process(target=frame_reader, args=(VIDEO_PATH, task_q))
    p_infer = Process(target=gpu_inference, args=(task_q, result_q, MODEL_WT))
    p_write = Process(target=video_writer_nvenc, args=(result_q, W, H, FPS))

    print(f"--- 启动 i9 + 3070Ti NVENC 并发引擎 ---")
    p_read.start()
    p_infer.start()
    p_write.start()

    p_read.join()
    p_infer.join()
    p_write.join()
    print(f"\n[{time.strftime('%H:%M:%S')}] 所有核心任务已完成！")