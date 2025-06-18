from RL.RLBases import RLInnerModelCallBack
from RL.common.utils.decorator.TorchMeta import SerializeAndTBMeta
class SyncTargetNetWorkStrategy(RLInnerModelCallBack, metaclass=SerializeAndTBMeta):
    '''
    使用policy_net更新target_net
    '''
