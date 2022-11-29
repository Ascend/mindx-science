from mindspore import nn


class Schrodinger_Net(nn.Cell):
    def __init__(self):
        super(Schrodinger_Net, self).__init__()
        self.fc1 = nn.Dense(2, 100, activation='tanh', weight_init='xavier_uniform')
        self.fc2 = nn.Dense(100, 100, activation='tanh', weight_init='xavier_uniform')
        self.fc3 = nn.Dense(100, 100, activation='tanh', weight_init='xavier_uniform')
        self.fc4 = nn.Dense(100, 100, activation='tanh', weight_init='xavier_uniform')
        self.fc5 = nn.Dense(100, 2, weight_init='xavier_uniform')

    def construct(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


def dict_transfer_Poisson(model, param_dict):
    convert_ckpt_dict = {}
    for _, param in model.parameters_and_names():
        convert_name1 = "jac2.model.model.cell_list." + param.name
        convert_name2 = "jac2.model.model.cell_list." + ".".join(param.name.split(".")[2:])
        for key in [convert_name1, convert_name2]:
            if key in param_dict:
                convert_ckpt_dict[param.name] = param_dict[key]
    return convert_ckpt_dict


def dict_transfer_NS(model, param_dict):
    convert_ckpt_dict = {}
    for _, param in model.parameters_and_names():
        convert_name1 = "jac2.model.model.cell_list." + param.name
        convert_name2 = "jac2.model.model.cell_list." + ".".join(param.name.split(".")[2:])
        for key in [convert_name1, convert_name2]:
            if key in param_dict:
                convert_ckpt_dict[param.name] = param_dict[key]
    return convert_ckpt_dict
