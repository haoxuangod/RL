from RL.common.syncTargetNetWorkStrategies import SyncTargetNetWorkStrategy


class EpisodeSyncTargetNetWorkStrategy(SyncTargetNetWorkStrategy):
    def __init__(self, episode_interval=5):
        self.episode_interval = episode_interval

    def onEpisodeFinished(self, episode, total_reward, model,env,is_eval=False):
        if is_eval:
            if episode % self.episode_interval == 0:
                model.target_net.load_state_dict(model.policy_net.state_dict())

    def onStepFinished(self, episode, state, next_state, action, reward, done, info, model,is_eval=False):
        pass
