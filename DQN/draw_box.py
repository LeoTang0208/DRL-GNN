import matplotlib.pyplot as plt
import os

f = open("results_orig.txt", "r") #_orig_diff_capa

drl = []
sap = []
lb = []

for line in f:
    a = []
    a.extend([float(i) for i in line.split()])
    drl.append(float(a[0] * 64))
    sap.append(float(a[1] * 64))
    lb.append(float(a[2] * 64))

f.close()

print(sum(drl) / len(drl), sum(sap) / len(sap))

plt.boxplot([drl, sap, lb], labels=['DRL+GNN', 'SAP', 'LB'])
plt.ylim([200, 1700])
plt.gca().yaxis.grid(True)

# plt.xlabel('Standard Deviation (\u03C3)', fontsize=12)
plt.ylabel('Bandwidth Allocated', fontsize=12)
    
plt.title('Tested on GEANT2', fontsize=12)

plt.show()