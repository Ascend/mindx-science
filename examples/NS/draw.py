# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import re

filename = "test09261545.txt"
prefix = "pictures/"

f = open(filename,"r")
data = f.read()

loss_module = re.compile("loss is (.*?)\n")
losses = [float(i) for i in loss_module.findall(data)]

l2_module = re.compile("l2_error\(u,v\):  (.*?)\n")
l2es = [float(i) for i in l2_module.findall(data)]

print(losses)

f.close()

f = open(filename,"r")
data = f.readlines()

C1, C2 = [], []
idx_c1 ,idx_c2 = 207,208

for i in range(200):
    C1.append(float(data[idx_c1 + i * 208][:-1]))
    C2.append(float(data[idx_c2 + i * 208][:-1]))

epoches = list(range(1,20001))

loss_epoches = [i*100 for i in list(range(1,201))]

var_epoches = [i*100 for i in list(range(1,201))]

plt.figure()

# l1 = plt.plot(epoches,losses,'g-',label="loss")
# l2 = plt.plot(loss_epoches,l2es,'b-',label="l2_error(u,v)")

# l_c1 = plt.plot(var_epoches,C1,'g-',label="C1_pred")
l_c2 = plt.plot(var_epoches,C2,'b-',label="C2_pred")
# l_c1_true = plt.plot(var_epoches,[1.0]*200,'r--',label="C1_true")
l_c2_true = plt.plot(var_epoches,[0.01]*200,'r--',label="C2_true")

# plt.ylim(0,3)

plt.legend()

plt.xlabel("epoch")
plt.ylabel("value")
plt.title("C2 in PINN solution of NS Equation")
plt.savefig(prefix + "C2-error-NS.png")

