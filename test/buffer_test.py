import time
import numpy as np
from collections import deque, namedtuple
'''
验证使用deque和namedtuple或每一个字段使用numpy数组分开维护谁速度快
'''
# Configuration
capacity = 100_000
state_dim = 10
batch_size = 64
n_runs = 1000

# Define transition namedtuple
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

# Generate random data for buffer
def random_transition():
    return Transition(
        state=np.random.randn(state_dim).astype(np.float32),
        action=np.random.randint(0, 4),
        reward=np.float32(np.random.rand()),
        next_state=np.random.randn(state_dim).astype(np.float32),
        done=bool(np.random.rand() > 0.95)
    )

# Populate buffers
deque_buf = deque(maxlen=capacity)
states_arr = np.zeros((capacity, state_dim), dtype=np.float32)
actions_arr = np.zeros(capacity, dtype=np.int64)
rewards_arr = np.zeros(capacity, dtype=np.float32)
next_states_arr = np.zeros((capacity, state_dim), dtype=np.float32)
dones_arr = np.zeros(capacity, dtype=bool)

for i in range(capacity):
    tr = random_transition()
    deque_buf.append(tr)
    states_arr[i] = tr.state
    actions_arr[i] = tr.action
    rewards_arr[i] = tr.reward
    next_states_arr[i] = tr.next_state
    dones_arr[i] = tr.done

# Pre-generate random indices for sampling
all_indices = np.random.randint(0, capacity, size=(n_runs, batch_size))

# Timing: deque + np.stack approach
t0 = time.perf_counter()
for run in range(n_runs):
    idx = all_indices[run]
    batch_states = np.stack([deque_buf[i].state for i in idx])
    batch_actions = np.array([deque_buf[i].action for i in idx])
    batch_rewards = np.array([deque_buf[i].reward for i in idx])
    batch_next_states = np.stack([deque_buf[i].next_state for i in idx])
    batch_dones = np.array([deque_buf[i].done for i in idx])
t1 = time.perf_counter()
deque_stack_time = t1 - t0

# Timing: preallocated numpy arrays + direct indexing
t0 = time.perf_counter()
for run in range(n_runs):
    idx = all_indices[run]
    batch_states = states_arr[idx]
    batch_actions = actions_arr[idx]
    batch_rewards = rewards_arr[idx]
    batch_next_states = next_states_arr[idx]
    batch_dones = dones_arr[idx]
t1 = time.perf_counter()
numpy_index_time = t1 - t0

# Report
print(f"Deque + np.stack total time for {n_runs} runs: {deque_stack_time:.4f}s")
print(f"NumPy array indexing total time for {n_runs} runs: {numpy_index_time:.4f}s")
print(f"Speedup: {deque_stack_time / numpy_index_time:.2f}x")
