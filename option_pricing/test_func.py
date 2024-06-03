import pandas as pd
import timeit


def greeks_stats(methods: dict) -> pd.DataFrame:
    data = {}
    index = []
    
    for func_name, greeks in methods.items():
        index.append(func_name)
        data.setdefault('Delta', []).append(greeks[0].item())
        data.setdefault('Gamma', []).append(greeks[1].item())
        data.setdefault('Vega', []).append(greeks[2].item())
        data.setdefault('Theta', []).append(greeks[3].item())
        data.setdefault('Rho', []).append(greeks[4].item())
    
    greekstat = pd.DataFrame(data, index=index)
    return greekstat


def time_stats(methods: dict, num_runs: int = 1000, num_repeats: int = 5) -> pd.DataFrame:
    data = {}
    index = []
    
    for func_name, method in methods.items():
        index.append(func_name)
        
        timer = timeit.Timer(method)
        times = timer.repeat(repeat=num_repeats, number=num_runs)
        avg_time_per_call = sum(times) / num_repeats / num_runs
        data.setdefault('Avg Time per Call (μs)', []).append(avg_time_per_call * 1e6)
    
    time_stat = pd.DataFrame(data, index=index)
    return time_stat

