import time

from utils import params
from SnowballOptionCalc import SnowBallOption

cpu = SnowBallOption(params, method='fdm', device='cpu')
gpu = SnowBallOption(params, method='fdm', device='gpu')


def compare_cpu_gpu(cpu, gpu):
    start_time = time.time()
    cpu_result = cpu.calc_opt_prc()
    cpu_time = time.time() - start_time

    start_time = time.time()
    gpu_result = gpu.calc_opt_prc()
    gpu_time = time.time() - start_time


    return cpu_time, gpu_time, cpu_result, gpu_result

cpu_time, gpu_time, cpu_result, gpu_result = compare_cpu_gpu(cpu, gpu)
T = params["T"]
numpath = params["trials"]

print('----------------------------------------')
print(f"定价方式: {cpu.get_method()}")
print(f"雪球期权存续期: {T}年")
print(f"模拟路径数量: {numpath}")
print("\n")
print(f"CPU 运行时间: {cpu_time} 秒")
print(f"CPU 计算结果: {cpu_result.round(4)}")
print(f"GPU 运行时间: {gpu_time} 秒")
print(f"GPU 计算结果: {gpu_result.round(4)}")
print('----------------------------------------')
