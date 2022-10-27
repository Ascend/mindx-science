# !usr/bin/env python
# -*- coding:utf-8 _*-

from mindelec.loss import Constraints as C

class Constraints(C):
    def __init__(self,dataset, pde_dict):
        super(Constraints, self).__init__(dataset, pde_dict)


