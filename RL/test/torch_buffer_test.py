import time
import numpy as np
import torch
'''
测试numpy抽取数据后转换为torch/直接使用torch索引
Numpy→Torch 总耗时: 0.4866s
Torch 索引   总耗时: 0.3315s
速度提升      : 1.47x
'''
def benchmark_numpy_to_torch(capacity, state_dim, batch_size, n_runs, device):
    # 生成随机 numpy 数据
    states_arr      = np.random.randn(capacity, state_dim).astype(np.float32)
    actions_arr     = np.random.randint(0, 4, size=capacity, dtype=np.int64)
    rewards_arr     = np.random.rand(capacity).astype(np.float32)
    next_states_arr = np.random.randn(capacity, state_dim).astype(np.float32)
    dones_arr       = (np.random.rand(capacity) > 0.95)

    # 预生成采样索引
    all_idx = np.random.randint(0, capacity, size=(n_runs, batch_size))

    # 测试：每次都 numpy → torch
    t0 = time.perf_counter()
    for run in range(n_runs):
        idx = all_idx[run]
        _ = torch.from_numpy(states_arr[idx]).to(device, non_blocking=True)
        _ = torch.from_numpy(actions_arr[idx]).to(device)
        _ = torch.from_numpy(rewards_arr[idx]).to(device)
        _ = torch.from_numpy(next_states_arr[idx]).to(device, non_blocking=True)
        _ = torch.from_numpy(dones_arr[idx].astype(np.uint8)).to(device)
    t1 = time.perf_counter()

    return t1 - t0

def benchmark_torch_index(capacity, state_dim, batch_size, n_runs, device):
    # 生成随机 numpy 数据并拷贝到预分配的 torch.Tensor
    states_arr      = np.random.randn(capacity, state_dim).astype(np.float32)
    actions_arr     = np.random.randint(0, 4, size=capacity, dtype=np.int64)
    rewards_arr     = np.random.rand(capacity).astype(np.float32)
    next_states_arr = np.random.randn(capacity, state_dim).astype(np.float32)
    dones_arr       = (np.random.rand(capacity) > 0.95)

    # 在目标 device 上预分配
    states_t   = torch.empty(capacity, state_dim, dtype=torch.float32, device=device)
    actions_t  = torch.empty(capacity,      dtype=torch.int64,   device=device)
    rewards_t  = torch.empty(capacity,      dtype=torch.float32, device=device)
    next_t     = torch.empty(capacity, state_dim, dtype=torch.float32, device=device)
    dones_t    = torch.empty(capacity,      dtype=torch.uint8,   device=device)

    # 只做一次拷贝
    states_t.copy_(torch.from_numpy(states_arr))
    actions_t.copy_(torch.from_numpy(actions_arr))
    rewards_t.copy_(torch.from_numpy(rewards_arr))
    next_t.copy_(torch.from_numpy(next_states_arr))
    dones_t.copy_(torch.from_numpy(dones_arr.astype(np.uint8)))

    # 预生成采样索引
    all_idx = np.random.randint(0, capacity, size=(n_runs, batch_size))

    # 测试：直接索引
    t0 = time.perf_counter()
    for run in range(n_runs):
        idx = all_idx[run]
        _ = states_t[idx]
        _ = actions_t[idx]
        _ = rewards_t[idx]
        _ = next_t[idx]
        _ = dones_t[idx]
    t1 = time.perf_counter()

    return t1 - t0

if __name__ == "__main__":
    capacity   = 100_000
    state_dim  = 10
    batch_size = 64
    n_runs     = 1_000
    device     = 'cuda'   # or 'cuda'

    t_np2t  = benchmark_numpy_to_torch(capacity, state_dim, batch_size, n_runs, device)
    t_tidx  = benchmark_torch_index(capacity, state_dim, batch_size, n_runs, device)
    speedup = t_np2t / t_tidx

    print(f"Numpy→Torch 总耗时: {t_np2t:.4f}s")
    print(f"Torch 索引   总耗时: {t_tidx:.4f}s")
    print(f"速度提升      : {speedup:.2f}x")