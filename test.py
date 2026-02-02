import torch
# 目标：输出必须包含 (12, 1)
print(f"当前可用算力: {torch.cuda.get_device_capability()}")