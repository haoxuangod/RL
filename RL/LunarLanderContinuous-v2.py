
import gym
import gymnasium.spaces
import torch

from RL.HierarchicalRL.HRLBases import HRLNode, Tree, HRLInnerModel
from RL.HierarchicalRL.HRLModels.HRLDQNModel import HRLNAFInnerModel
from RL.HierarchicalRL.HRLModels.IntraOption import IntraOptionDQNInnerModel, IntraOptionNAFInnerModel, IntraOptionNode, \
    IntraOptionDoubleDQNInnerModel, IntraOptionDoubleNAFInnerModel

from RL.RLBases import EpsilonGreedy, RLAgent, OUNoise
from RL.RLModels.ActorCriticOnPolicyInnerModel import ActorCriticOnPolicyInnerModel

from RL.RLModels.ValueOffPolicyInnerModel.DQNModel import StepSyncTargetNetWorkStrategy, TimeSeriesNAFInnerModel, \
    ConservativeDoubleNAFInnerModel, NAFInnerModel, PopArtNAFInnerModel

'''
任务概述
LunarLanderContinuous-v2 是一个基于 Box2D 的二维火箭着陆仿真任务。智能体需要通过像素或物理状态信息驱动
两个发动机（主发动机和横向姿态发动机），使火箭着陆器（Lander）平稳降落到位于(0,0)的着陆平台上，同时尽量避免
碰撞或飞出视野。环境会给出连续的状态反馈和稠密的奖励，鼓励迅速且平稳地完成着陆任务。

如果一个 episode 的总累计奖励达到200 分，环境即视为被“解决”（solved）。

状态（Observation）空间
    状态空间是一个8维向量，包含以下元素：x、y坐标，x、y速度，角度，角速度，左腿接触，右腿接触。
    类型：`Box`
        形状：(8,)
        取值范围：
            low  = [-1.5, -1.5, -5.0, -5.0, -3.14, -5.0, 0.0, 0.0]
            high = [ 1.5,  1.5,  5.0,  5.0,  3.14,  5.0, 1.0, 1.0]
        数据类型：`float32`
        含义：八维向量，依次表示
        1. 水平位置 x
        2. 竖直位置 y
        3. 水平速度 vx
        4. 竖直速度 vy
        5. 旋转角度 θ
        6. 角速度 w
        7. 左支腿接触地面标志（0 或 1）
        8. 右支腿接触地面标志（0 或 1）
动作（Action）空间
    动作空间是一个范围在[-1, 1]之间的2维连续数组，表示主引擎和侧推力的强度。
    类型：`Box`
    形状：(2,)
    取值范围：[-1.0, +1.0]
    数据类型：`float32`
    含义：长度为 2 的向量 [main,lateral]，分别控制
        1. 主发动机推力（main）：
            当 main < 0 时，主发动机关闭；
            当 0<=main<=1 时，推力从 50% 线性增至 100%。
        2. 横向姿态发动机（lateral）：
            当 -0.5 < lateral < 0.5 时，不喷射任何侧向推力；
            当 lateral < -0.5 时，左侧姿态发动机喷射，推力从 50% 线性增至 100%；
            当 lateral > 0.5 时， 右侧姿态发动机喷射，推力同样从 50% 线性增至 100%


'''


env = gym.make("LunarLanderContinuous-v2", render_mode="rgb_array")


def HRL():
    # 环境配置
    state_dim = env.observation_space.shape
    print("state_shape:", env.observation_space.shape)
    print("state_type:",type(env.reset()[0]))
    print("action_shape:", env.action_space)
    strategy = EpsilonGreedy()
    #strategy = Boltzmann()
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root_inner_model=IntraOptionDoubleDQNInnerModel(env.observation_space,2,device=device ,memory_capacity=500000,n_steps=5,
                                        exploration_strategy=strategy, sync_target_strategy=StepSyncTargetNetWorkStrategy(steps_interval=1000))
    node1_inner_model = IntraOptionNAFInnerModel(env.observation_space, env.action_space, device=device, memory_capacity=500000,
                                        n_steps=5,
                                        sync_target_strategy=StepSyncTargetNetWorkStrategy(steps_interval=1000))
    node2_inner_model = IntraOptionNAFInnerModel(env.observation_space, env.action_space, device=device, memory_capacity=500000,
                                         n_steps=5,
                                         sync_target_strategy=StepSyncTargetNetWorkStrategy(steps_interval=1000))
    root=IntraOptionNode("Root",root_inner_model)
    node1=IntraOptionNode("Node1",node1_inner_model)
    node2=IntraOptionNode("Node2",node2_inner_model)
    root.subtasks=[node1,node2]
    tree=Tree()
    tree.load_from_root(root)
    inner_model = HRLInnerModel(state_space=env.observation_space,action_space=env.action_space,tree=tree,gamma=0.99, memory_capacity=10000,)
    agent = RLAgent("RLAgent", env, inner_model, render_mode=1,directory="LunarLanderContinuous-v2",save_interval=100)
    #agent.load()
    agent.train()
    agent.draw_history("LunarLanderContinuous-v2_IntraOption.png")

def HRL_one_node():
    state_dim = env.observation_space.shape
    print("state_shape:", env.observation_space.shape)
    print("state_type:", type(env.reset()[0]))
    print("action_shape:", env.action_space)
    strategy = EpsilonGreedy()
    # strategy = Boltzmann()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root_inner_model = HRLNAFInnerModel(env.observation_space, env.action_space, device=device, memory_capacity=500000,
                                         n_steps=5,
                                         exploration_strategy=strategy,
                                         sync_target_strategy=StepSyncTargetNetWorkStrategy(steps_interval=1000))
    root = HRLNode("Root", root_inner_model)

    tree = Tree()
    tree.load_from_root(root)
    inner_model = HRLInnerModel(state_space=env.observation_space, action_space=env.action_space, tree=tree, gamma=0.99,
                                exploration_strategy=strategy, memory_capacity=10000, )
    agent = RLAgent("RLAgent", env, inner_model, render_mode=1, directory="LunarLanderContinuous-v2", save_interval=100)
    # agent.load()
    agent.train()
    agent.draw_history("LunarLanderContinuous-v2.png")

def stable_baselines3():
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

    def make_env():
        """
        Create and wrap the LunarLanderContinuous-v2 environment.
        """
        env = gym.make("LunarLanderContinuous-v3")
        env = Monitor(env)
        return env

    # Vectorized environments for training
    env = DummyVecEnv([make_env])
    # Normalize observations and rewards for stable training
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Separate evaluation environment (no reward normalization)
    eval_env = DummyVecEnv([lambda: Monitor(gym.make("LunarLanderContinuous-v3"))])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # Callback to stop training once the agent reaches a reward threshold
    early_stop = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=early_stop,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/",
        eval_freq=10000,
        n_eval_episodes=10,
        verbose=1
    )

    # Instantiate the PPO model with chosen hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1
    )

    # Train the agent
    model.learn(total_timesteps=500_000, callback=eval_callback)

    # Save the trained model
    model.save("ppo_lunarlander_continuous")

    # ===== Evaluation and Rendering =====
    # Load the best saved model if desired
    # model = PPO.load("./logs/best_model/best_model", env=env)

    # Evaluate and render
    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()
def one_model():
    print("state:", env.observation_space.shape)
    print(env.observation_space)
    print("state_type:", type(env.reset()[0]))
    print("action_shape:", env.action_space)
    # strategy = Boltzmann()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    inner_model = TimeSeriesNAFInnerModel(env.observation_space, env.action_space, device=device,window_size=7,memory_capacity=500000,
                                        n_steps=1,
                                        sync_target_strategy=StepSyncTargetNetWorkStrategy(steps_interval=1000))
    '''
    inner_model=ActorCriticOnPolicyInnerModel(env.observation_space, env.action_space, device=device,t_steps=5,
                                              gamma=0.99,ent_coef=0.01)
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from tensorboard import program
    tb = program.TensorBoard()
    tensorboard_dir = './tensorboard_record_lunar_0/'
    tb.configure(argv=[None, '--logdir', tensorboard_dir, '--port', '6006'])
    url = tb.launch()
    print(f"TensorBoard listening on {url}")
    inner_model = ConservativeDoubleNAFInnerModel(env.observation_space,env.action_space,  device=device,
                                          memory_capacity=60000,
                                          n_steps=5, update_freq=4,steps_before_update=1,record_stats_dir=tensorboard_dir,
                                          lr=1e-4,exploration_strategy=OUNoise(action_dimension=2))

    inner_model = PopArtNAFInnerModel(env.observation_space, env.action_space, device=device,
                                                  memory_capacity=60000,
                                                  n_steps=1, update_freq=4, steps_before_update=1,
                                                  record_stats_dir=tensorboard_dir,
                                                  lr=1e-4)
    

    agent = RLAgent("RLAgent", env, inner_model, render_mode=1, directory="LunarLanderContinuous-v2",
                    max_iter=10000,save_interval=100)
    # agent.load()
    agent.train()
    agent.draw_history("A2C_LunarLanderContinuous-v2.png")

if __name__ == "__main__":
    one_model()
    #HRL()
    #HRL_one_node()
    #stable_baselines3()