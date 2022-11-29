import json
import time
import sys
import torch
from torch import Tensor
from os.path import abspath, dirname

path = abspath(dirname(dirname(__file__)))
sys.path.append(path)
from base.FNN import FNN
from base.data_process import get_Poisson_data, get_Schrodinger_data, get_NS_data
from base.error_calculation import Poisson_error, Schrodinger_error, NS_error

configs = json.load(open("./config.json"))

# create models
Poisson_deepxde = FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
Schrodinger_deepxde = FNN([2] + [100] * 5 + [2], "tanh", "Glorot normal")
NS_deepxde = FNN([3] + [50] * 6 + [3], "tanh", "Glorot uniform")

# load models
Poisson_deepxde.load_state_dict(torch.load(configs["Poisson_pth_path"]))
Poisson_deepxde.eval()
Schrodinger_deepxde.load_state_dict(torch.load(configs["Schrodinger_pth_path"]))
Schrodinger_deepxde.eval()
NS_deepxde.load_state_dict(torch.load(configs["NS_pth_path"]))
NS_deepxde.eval()

# data preprocess
Poisson_inputs, Poisson_labels = get_Poisson_data(configs["Poisson_data_path"])
Poisson_inputs = Tensor(Poisson_inputs)
Schrodinger_inputs, Schrodinger_labels = get_Schrodinger_data(configs["Schrodinger_data_path"])
Schrodinger_inputs = Tensor(Schrodinger_inputs)
NS_inputs, NS_labels = get_NS_data(configs["NS_data_path"], 700000)
NS_inputs = Tensor(NS_inputs)

# infer
tp, ts, tn = [], [], []
for i in range(100):
    T1 = time.time()
    Poisson_results = Poisson_deepxde(Poisson_inputs)
    Poisson_time = time.time() - T1
    if i >= 50:
        tp.append(Poisson_time)

for i in range(100):
    T1 = time.time()
    Schrodinger_results = Schrodinger_deepxde(Schrodinger_inputs)
    Schrodinger_time = time.time() - T1
    if i >= 50:
        ts.append(Schrodinger_time)

for i in range(100):
    T1 = time.time()
    NS_results = NS_deepxde(NS_inputs)
    NS_time = time.time() - T1
    if i >= 50:
        tn.append(NS_time)

# postprocess
Poisson_L2 = Poisson_error(Poisson_results.detach().numpy(), Poisson_labels)
Schrodinger_L2 = Schrodinger_error(Schrodinger_results.detach().numpy(), Schrodinger_labels)
u_L2, v_L2, p_L2 = NS_error(NS_results.detach().numpy(), NS_labels)

# print
print("Results of online inference for deepxde(baseline)")
print("Performance")
print("For Poisson: the time is {0}".format(sum(tp) / len(tp)))
print("For Schrodinger: the time is {0}".format(sum(ts) / len(ts)))
print("For NS: the time is {0}".format(sum(tn) / len(tn)))
print("Accuracy")
print("For Poisson: the error is {0}".format(Poisson_L2.item()))
print("For Schrodinger: the error is {0}".format(Schrodinger_L2.item()))
print("For NS: the u/v errors are {0}, {1}".format(u_L2.item(), v_L2.item()))
