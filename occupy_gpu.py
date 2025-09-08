import torch
import time
import random

# 检查是否有可用的 GPU
if not torch.cuda.is_available():
    raise RuntimeError("GPU 不可用，请检查你的环境！")

# 获取所有可用 GPU 的设备列表
device_ids = list(range(torch.cuda.device_count()))
print(f"检测到的 GPU 数量: {len(device_ids)}")

# 获取每个 GPU 的总显存
device_properties = [torch.cuda.get_device_properties(i) for i in device_ids]
total_memory = min([prop.total_memory for prop in device_properties])  # 选取最小显存的 GPU 避免超出

print(total_memory)

# 转换为适合矩阵存储的大小（float32，每个元素占用 4 字节）
element_size = 4  # 字节
matrix_size = int((total_memory * 0.9) // element_size)**0.5 * 0.5 # 使用 30% 显存
print(f"每块 GPU 上的矩阵维度: {int(matrix_size)}x{int(matrix_size)}")

# 创建矩阵并分发到所有 GPU
A_split = [torch.randn((int(matrix_size), int(matrix_size)), device=f"cuda:{i}") for i in device_ids]
B_split = [torch.randn((int(matrix_size), int(matrix_size)), device=f"cuda:{i}") for i in device_ids]

# 循环执行
try:
    while True:
        print("开始矩阵乘法...")

        # 在每个 GPU 上执行矩阵乘法
        results = []
        for i, (A_gpu, B_gpu) in enumerate(zip(A_split, B_split)):
            print(f"在 GPU {i} 上进行计算...")
            C = torch.matmul(A_gpu, B_gpu)
            results.append(C)  # 将结果存储在列表中

        print("所有 GPU 上的矩阵乘法完成！")

        # 清理结果并释放显存
        del results
        torch.cuda.empty_cache()

        # 随机休息 1 到 5 秒
        sleep_time = random.uniform(5, 10)
        print(f"休息 {sleep_time:.2f} 秒...")
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("程序被用户中断，退出...")

