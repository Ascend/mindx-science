# !usr/bin/env python
# -*- coding:utf-8 _*-


from mindelec.loss import Constraints
#
# class SupervisedConstraints(Constraints):
#     def __init__(self, dataset, pde_dict):
#         super(SupervisedConstraints, self).__init__(dataset, pde_dict)
#         self.supervised_dict = dataset.supervised_dict
#         self.train_data_list = dataset.train_data_list


class SupervisedConstraints(Constraints):
    def __init__(self, dataset, pde_dict):
        super(SupervisedConstraints, self).__init__(dataset, pde_dict)
        self.label_data = dataset.label_data


