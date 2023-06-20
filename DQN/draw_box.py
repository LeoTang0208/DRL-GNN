import matplotlib.pyplot as plt
import os

f = open("./result_logs/results_NSFnet_plr.txt", "r") #_orig_diff_capa

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

print(sum(drl) / len(drl), sum(sap) / len(sap))

plt.boxplot([drl, sap, lb], labels=['DRL+GNN', 'SAP', 'LB'])
plt.ylim([100, 1500])
plt.gca().yaxis.grid(True)

# plt.xlabel('Standard Deviation (\u03C3)', fontsize=12)
plt.ylabel('Bandwidth Allocated', fontsize=12)
    
plt.title('Tested on NSFnet', fontsize=12)

plt.show()