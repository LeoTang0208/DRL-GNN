import matplotlib.pyplot as plt
import os

f = open("./result_logs/results_GEANT2_plr.txt", "r") #_orig_diff_capa

drl = []
sap = []
lb = []

for line in f:
    a = []
    a.extend([float(i) for i in line.split()])
    drl.append(float(a[2] * 64))
    sap.append(float(a[3] * 64))
    lb.append(float(a[4] * 64))

f.close()

# print(sum(drl) / len(drl), sum(sap) / len(sap))

plt.boxplot([drl, sap, lb], labels=['DRL+GNN', 'SAP', 'LB'])
plt.ylim([000, 1600])
plt.gca().yaxis.grid(True)

# plt.xlabel('Standard Deviation (\u03C3)', fontsize=12)
plt.ylabel('Bandwidth Allocated', fontsize=12)
    
plt.title('Tested on GEANT2', fontsize=12)

plt.show()