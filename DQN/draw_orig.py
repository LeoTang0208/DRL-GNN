import matplotlib.pyplot as plt
import os

f = open("results_orig_diff_capa.txt", "r")

x = []
drl = []
sap = []
lb = []

cnt = 0
for line in f:
    a = []
    a.extend([float(i) for i in line.split()])
    x.append(float(a[0]))
    drl.append(float(a[1] * 64))
    sap.append(float(a[2] * 64))
    lb.append(float(a[3] * 64))

# data = [drl, sap, lb]
# bp = plt.boxplot(data)
# plt.xticks([1, 2, 3], ['DRL+GNN', 'SAP', 'LB'])
# plt.grid(axis="y")
# plt.show()

f.close()

x_m = [0, 0.5, 1, 1.5, 2, 2.5, 3]
drl_m = []
sap_m = []
lb_m = []
for x_ in x_m:
    t = 0
    s1 = float(0)
    s2 = float(0)
    s3 = float(0)
    
    f = open("results_orig_diff_capa.txt", "r")
    for line in f:
        a = []
        a.extend([float(i) for i in line.split()])
        if (a[0] == x_):
            s1 = s1 + a[1]
            s2 = s2 + a[2]
            s3 = s3 + a[3]
            t = t + 1

    drl_m.append(float(s1 / t * 64))
    sap_m.append(float(s2 / t * 64))
    lb_m.append(float(s3 / t * 64))

    f.close()

d1 = plt.scatter(x, drl, c="r", s=5, alpha=0.5)
d2 = plt.scatter(x, sap, c="g", s=5, alpha=0.5)
d3 = plt.scatter(x, lb, c="b", s=5, alpha=0.5)

l1 = plt.plot(x_m, drl_m, "r*-", linewidth=2, label="DRL")
l2 = plt.plot(x_m, sap_m, "g*-", linewidth=2, label="SAP")
l3 = plt.plot(x_m, lb_m, "b*-", linewidth=2, label="LB")

plt.grid()
plt.legend(loc="upper right")

# plt.xlim([-0.1, 3.1])
# plt.ylim([6, 18])

plt.xlabel('Standard Deviation (\u03C3)', fontsize=12)
plt.ylabel('Bandwidth Allocated', fontsize=12)

plt.show()