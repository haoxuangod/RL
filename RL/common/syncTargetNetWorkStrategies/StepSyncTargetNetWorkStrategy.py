from RL.common.syncTargetNetWorkStrategies import SyncTargetNetWorkStrategy
class StepSyncTargetNetWorkStrategy(SyncTargetNetWorkStrategy):
    def __init__(self, steps_interval=1000):
        self.steps_interval = steps_interval
        self.steps = 0

    def onEpisodeFinished(self, episode, total_reward, model, env, is_eval=False):
        pass

    def onStepFinished(self, episode, state, next_state, action, reward, done, info, model,is_eval=False):
        if not is_eval:
            self.steps = self.steps + 1
            if self.steps % self.steps_interval == 0:
                model.target_net.load_state_dict(model.policy_net.state_dict())
