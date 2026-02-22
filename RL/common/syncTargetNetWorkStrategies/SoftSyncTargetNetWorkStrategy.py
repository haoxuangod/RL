from RL.common.syncTargetNetWorkStrategies import SyncTargetNetWorkStrategy


class SoftSyncTargetNetWorkStrategy(SyncTargetNetWorkStrategy):
    def __init__(self, tau):
        self.tau = tau

    def onEpisodeFinished(self, episode, total_reward, model,env,is_eval=False):
        pass

    def onUpdateFinished(self,update_cnt,model):
        for target_param, policy_param in zip(
                model.target_net.parameters(), model.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(self.base_tag+"/test",1,0)
        '''
        model.target_net.output_net[0].popart.mu = model.policy_net.output_net[0].popart.mu
        model.target_net.output_net[0].popart.sigma = model.policy_net.output_net[0].popart.sigma
        model.target_net.output_net[1].popart.mu = model.policy_net.output_net[1].popart.mu
        model.target_net.output_net[1].popart.sigma = model.policy_net.output_net[1].popart.sigma
        '''
    '''
    def onStepFinished(self, episode, state, next_state, action, reward, done, info, model,is_eval=False):
        for target_param, policy_param in zip(
                model.target_net.parameters(), model.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )
        for target_param, policy_param in zip(
                model.target_net1.parameters(), model.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )
    '''

