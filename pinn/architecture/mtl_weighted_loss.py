# !usr/bin/env python
# -*- coding:utf-8 _*-

from mindelec.architecture import MTLWeightedLossCell as MWL

class MTLWeightedLossCell(MWL):
    def __init__(self,num_losses):
        super(MTLWeightedLossCell, self).__init__(num_losses)

    
