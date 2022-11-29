import json
import sys
import torch
from os.path import abspath, dirname

path = abspath(dirname(dirname(dirname(__file__))))
sys.path.append(path)
from base.FNN import FNN

configs = json.load(open("./config.json"))

# create models
Poisson_deepxde = FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
Schrodinger_deepxde = FNN([2] + [100] * 5 + [2], "tanh", "Glorot normal")
NS_deepxde = FNN([3] + [50] * 6 + [3], "tanh", "Glorot uniform")

# load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Poisson_deepxde.load_state_dict(torch.load(configs["Poisson_pth_path"], map_location=device))
Poisson_deepxde.eval()
Schrodinger_deepxde.load_state_dict(torch.load(configs["Schrodinger_pth_path"], map_location=device))
Schrodinger_deepxde.eval()
NS_deepxde.load_state_dict(torch.load(configs["NS_pth_path"], map_location=device))
NS_deepxde.eval()

# create onnx
torch.onnx.export(Poisson_deepxde, torch.randn(19521, 2), configs["Poisson_onnx_deepxde"],
                  opset_version=11, verbose=True, input_names=['inputs'], output_names=['outputs'])
torch.onnx.export(Schrodinger_deepxde, torch.randn(51456, 2), configs["Schrodinger_onnx_deepxde"],
                  opset_version=11, verbose=True, input_names=['inputs'], output_names=['outputs'])
torch.onnx.export(NS_deepxde, torch.randn(355000, 3), configs["NS_onnx_deepxde"],
                  opset_version=11, verbose=True, input_names=['inputs'], output_names=['outputs'])
