import matplotlib.pyplot as plt
import os

f = open("results_random_graph.txt", "r")

num = []

for line in f:
    a = []
    a.extend([float(i) for i in line.split()])
    num.append(a)

cnt = 1
x = []
drl = []
sap = []
lb = []
for i in range(len(num)):
    x.append(num[i][0])
    drl.append(num[i][2])
    sap.append(num[i][3])
    lb.append(num[i][4])

x_m = [10, 20, 30, 40]
drl_m = []
sap_m = []
lb_m = []
for x_ in x_m:
    t = 0
    s1 = float(0)
    s2 = float(0)
    s3 = float(0)
    for i in range(len(num)):
        if (num[i][0] == x_):
            s1 = s1 + num[i][2]
            s2 = s2 + num[i][3]
            s3 = s3 + num[i][4]
            t = t + 1

    drl_m.append(float(s1 / t))
    sap_m.append(float(s2 / t))
    lb_m.append(float(s3 / t))

plt.subplot(1, 2, 1)
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

plt.xlabel('Size', fontsize=12)
plt.ylabel('Score', fontsize=14)

# plt.title('K = {} Shortest Paths Considered'.format(k))

plt.subplot(1, 2, 2)
diff_drl = []
diff_lb = []
for i in range(len(x_m)):
    diff_drl.append(float(drl_m[i] / sap_m[i]))
    diff_lb.append(float(lb_m[i] / sap_m[i]))

l4 = plt.plot(x_m, diff_drl, label="DRL")
l5 = plt.plot(x_m, diff_lb, label="LB")
plt.ylim([0.0, 1.0])
plt.grid()
plt.legend(loc="upper right")

plt.show()

#NETWORK DESIGN PROBLEM: where to allocate bdwdth
#Gods eye view
#REPRODUCE the box plot