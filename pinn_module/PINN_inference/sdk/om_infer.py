import numpy as np
from mindx.sdk.base import Tensor, Model
import json
import sys
from os.path import abspath, dirname

path = abspath(dirname(dirname(__file__)))
sys.path.append(path)
from base.data_process import get_Poisson_data, get_Schrodinger_data, get_NS_data
from base.error_calculation import Poisson_error, Schrodinger_error, NS_error

device_id = 0

configs = json.load(open("./config.json"))
Poisson = Model(configs["Poisson_om_path"], device_id)
Schrodinger = Model(configs["Schrodinger_om_path"], device_id)
NS = Model(configs["NS_om_path"], device_id)

'''preprocess'''
# Poisson
Poisson_inputs, Poisson_labels = get_Poisson_data(configs["Poisson_data_path"])
Poisson_tensor = Tensor(np.array(Poisson_inputs, dtype=np.float32))
Poisson_tensor.to_device(device_id)
Poisson_tensors = [Poisson_tensor]

# Schrodinger
Schrodinger_inputs, Schrodinger_labels = get_Schrodinger_data(configs["Schrodinger_data_path"])
Schrodinger_inputs = np.ascontiguousarray(np.array(Schrodinger_inputs, dtype=np.float32))
Schrodinger_tensor = Tensor(Schrodinger_inputs)
Schrodinger_tensor.to_device(device_id)
Schrodinger_tensors = [Schrodinger_tensor]

# NS
NS_inputs, NS_labels = get_NS_data(configs["NS_data_path"], 700000)
NS_tensor = Tensor(np.array(NS_inputs, dtype=np.float32))
NS_tensor.to_device(device_id)
NS_tensors = [NS_tensor]

'''infer_performance'''
# Poisson
Poisson_outputs = Poisson.infer(Poisson_tensors)
for i in range(len(Poisson_outputs)):
    Poisson_outputs[i].to_host()
Poisson_results = Poisson_outputs[0]
# Schrodinger
Schrodinger_outputs = Schrodinger.infer(Schrodinger_tensors)
for i in range(len(Schrodinger_outputs)):
    Schrodinger_outputs[i].to_host()
Schrodinger_results = Schrodinger_outputs[0]
# NS
NS_outputs = NS.infer(NS_tensors)
for i in range(len(NS_outputs)):
    NS_outputs[i].to_host()
NS_results = NS_outputs[0]

'''postprocess'''
Poisson_L2 = Poisson_error(np.array(Poisson_results), Poisson_labels)
Schrodinger_L2 = Schrodinger_error(np.array(Schrodinger_results), Schrodinger_labels)
u_L2, v_L2 = NS_error(np.array(NS_results), NS_labels)

'''print'''
print("Results of offline inference for pinn")
print("For Poisson: the error is {0}".format(Poisson_L2.item()))
print("For Schrodinger: the error is {0}".format(Schrodinger_L2.item()))
print("For NS: the u/v errors are {0}, {1}".format(u_L2.item(), v_L2.item()))
