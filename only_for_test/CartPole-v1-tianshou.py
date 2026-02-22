import gymnasium as gym
import torch
import tianshou as ts

from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import CollectStats
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.utils.space_info import SpaceInfo

from torch.utils.tensorboard import SummaryWriter


def main():
    task = "CartPole-v1"

    # ====== 超参数（你可以后面再调）======
    lr = 1e-3
    max_epochs = 10
    batch_size = 64

    num_training_envs = 1   # 关键：只用 1 个 env
    num_test_envs = 1       # 测试也用 1 个 env

    gamma = 0.9
    n_step = 3
    target_update_freq = 320

    buffer_size = 20000
    epoch_num_steps = 10000
    collection_step_num_env_steps = 10

    eps_train = 0.1
    eps_test = 0.05

    # ====== Logger ======
    logger = ts.utils.TensorboardLogger(SummaryWriter("log/dqn_cartpole_single_v2"))

    # ====== 环境：DummyVectorEnv 但只有 1 个环境（非并行）======
    training_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(num_training_envs)])
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(num_test_envs)])

    # 用一个普通 env 来读取空间信息（obs/action 形状）
    env = gym.make(task)
    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape

    # ====== Q 网络 ======
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])

    # ====== Policy + Algorithm ======
    policy = DiscreteQLearningPolicy(
        model=net,
        action_space=env.action_space,
        eps_training=eps_train,
        eps_inference=eps_test,
    )

    algorithm = ts.algorithm.DQN(
        policy=policy,
        optim=AdamOptimizerFactory(lr=lr),
        gamma=gamma,
        n_step_return_horizon=n_step,
        target_update_freq=target_update_freq,
    )

    # ====== Collector + Buffer ======
    training_collector = ts.data.Collector[CollectStats](
        algorithm,
        training_envs,
        ts.data.VectorReplayBuffer(buffer_size, num_training_envs),  # num_training_envs=1
        exploration_noise=True,
    )
    test_collector = ts.data.Collector[CollectStats](
        algorithm,
        test_envs,
        exploration_noise=True,
    )
    algorithm.policy.eps_training = 1.0  # DQN 常见：先全随机

    training_collector.collect(
        n_step=5000,  # ✅ 预采集 5000 个 env steps
        reset_before_collect=True  # ✅ 强制先 reset，避免初始 obs/info 为 None
    )

    algorithm.policy.eps_training = eps_train  # 再把训练 epsilon 调回正常值
    # ====== Stop 条件（达到环境 reward_threshold 就停）======
    def stop_fn(mean_rewards: float) -> bool:
        if env.spec and env.spec.reward_threshold is not None:
            return mean_rewards >= env.spec.reward_threshold
        return False

    # ====== 训练 ======
    result = algorithm.run_training(
        OffPolicyTrainerParams(
            training_collector=training_collector,
            test_collector=test_collector,
            max_epochs=max_epochs,
            epoch_num_steps=epoch_num_steps,
            collection_step_num_env_steps=collection_step_num_env_steps,
            test_step_num_episodes=10,  # 测 10 个 episode
            batch_size=batch_size,
            update_step_num_gradient_steps_per_sample=1 / collection_step_num_env_steps,
            stop_fn=stop_fn,
            logger=logger,
            test_in_training=True,
        )
    )
    print(f"Finished training in {result.timing.total_time:.2f}s")

    # ====== 观看（渲染）======
    watch_env = gym.make(task, render_mode="human")
    watcher = ts.data.Collector[CollectStats](algorithm, watch_env, exploration_noise=True)
    watcher.reset()
    watcher.collect(n_episode=5, render=1 / 35)


if __name__ == "__main__":
    main()
