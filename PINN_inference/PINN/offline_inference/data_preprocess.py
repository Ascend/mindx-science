import json
import sys
import numpy as np
from os.path import abspath, dirname

path = abspath(dirname(dirname(__file__)))
sys.path.append(path)

from base.data_process import get_Poisson_data, get_Schrodinger_data, get_NS_data

configs = json.load(open("./config.json"))

Poisson_inputs, _ = get_Poisson_data(configs["Poisson_data_path"])
np.ascontiguousarray(Poisson_inputs).tofile(configs["Poisson_bin_path"])
Schrodinger_inputs, _ = get_Schrodinger_data(configs["Schrodinger_data_path"])
np.ascontiguousarray(Schrodinger_inputs).tofile(configs["Schrodinger_bin_path"])
NS_inputs, _ = get_NS_data(configs["NS_data_path"], 700000)
np.ascontiguousarray(NS_inputs).tofile(configs["NS_bin_path"])
