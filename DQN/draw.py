import matplotlib.pyplot as plt
import os

f = open("./result_logs/results.txt", "r") #_orig_diff_capa

num = []

for line in f:
    a = []
    a.extend([float(i) for i in line.split()])
    num.append(a)

cnt = 1
for k in [4, 6, 8, 10]:
    x = []
    drl = []
    sap = []
    lb = []
    for i in range(len(num)):
        if (num[i][1] == k):
            x.append(num[i][0])
            drl.append(float(num[i][2] * 64))
            sap.append(float(num[i][3] * 64))
            lb.append(float(num[i][4] * 64))
    
    x_m = [0, 0.5, 1, 1.5, 2, 2.5, 3]
    drl_m = []
    sap_m = []
    lb_m = []
    for x_ in x_m:
        t = 0
        s1 = float(0)
        s2 = float(0)
        s3 = float(0)
        for i in range(len(num)):
            if ((num[i][1] == k) and (num[i][0] == x_)):
                s1 = s1 + float(num[i][2] * 64)
                s2 = s2 + float(num[i][3] * 64)
                s3 = s3 + float(num[i][4] * 64)
                t = t + 1
        
        drl_m.append(float(s1 / t))
        sap_m.append(float(s2 / t))
        lb_m.append(float(s3 / t))
    
    # plt.subplot(2, 2, cnt)
    fig = plt.figure(cnt)
    d1 = plt.scatter(x, drl, c = "r", s=5, alpha=0.5)
    d2 = plt.scatter(x, sap, c = "g", s=5, alpha=0.5)
    d3 = plt.scatter(x, lb, c = "b", s=5, alpha=0.5)
    
    l1 = plt.plot(x_m, drl_m, "r*-", linewidth=2, label="DRL")
    l2 = plt.plot(x_m, sap_m, "g*-", linewidth=2, label="SAP")
    l3 = plt.plot(x_m, lb_m, "b*-", linewidth=2, label="LB")
    
    plt.grid()
    
    plt.legend(loc = "upper right")
    
    plt.xlim([-0.1, 3.1])
    plt.ylim([300, 1100])
    
    plt.xlabel('Standard Deviation Indicator (\u03B3)', fontsize=12)
    plt.ylabel('Bandwidth Allocated', fontsize=12)
    
    plt.title('K = {} Shortest Paths Considered'.format(k))
    
    cnt = cnt + 1

# print(len(num))
plt.show()