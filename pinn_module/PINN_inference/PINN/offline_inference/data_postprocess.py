import json
import sys
from os.path import abspath, dirname
import numpy as np

path = abspath(dirname(dirname(__file__)))
sys.path.append(path)
from base.data_process import get_Poisson_data, get_Schrodinger_data, get_NS_data
from base.error_calculation import Poisson_error, Schrodinger_error, NS_error

configs = json.load(open("./config.json"))
Poisson_results = np.loadtxt(configs["Poisson_result_path"])
Schrodinger_results = np.loadtxt(configs["Schrodinger_result_path"])
NS_results = np.loadtxt(configs["NS_result_path"])

_, Poisson_labels = get_Poisson_data(configs["Poisson_data_path"])
_, Schrodinger_labels = get_Schrodinger_data(configs["Schrodinger_data_path"])
_, NS_labels = get_NS_data(configs["NS_data_path"], 700000)

Poisson_L2 = Poisson_error(Poisson_results.reshape(-1, 1), Poisson_labels)
Schrodinger_L2 = Schrodinger_error(Schrodinger_results, Schrodinger_labels)
u_L2, v_L2 = NS_error(NS_results, NS_labels)

print("Poisson error=", Poisson_L2)
print("Schrodinger error=", Schrodinger_L2)
print("NS error: u={0}, v={1}".format(u_L2, v_L2))
