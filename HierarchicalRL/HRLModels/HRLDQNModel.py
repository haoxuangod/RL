from RL.HierarchicalRL.HRLBases import HRLNodeInnerModelDecorator
from RL.RLModels.ValueOffPolicyInnerModel.DQNModel import DQNInnerModel, NAFInnerModel

class HRLDQNInnerModel(HRLNodeInnerModelDecorator(DQNInnerModel)):
   pass


class HRLNAFInnerModel(HRLNodeInnerModelDecorator(NAFInnerModel)):
    pass
