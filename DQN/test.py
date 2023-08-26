import subprocess
import random

random.seed()

cmd_base = "python evaluate_DQN.py -d ./Logs/expsample_DQN_agent_plr_4_Logs.txt"

# for p in range(11):
#     if (p != 0):
#         for j in range(5):
#             seed = random.randint(0, 50)
#             cmd = cmd_base + " -s " + str(0) + " -e " + str(0) + " -p " + str(p / 10) + " -v " + str(0.0)
#             subprocess.call(cmd, shell=True)
#             print(">>>>> ", p, j)
#     else:
#         seed = random.randint(0, 50)
#         cmd = cmd_base + " -s " + str(0) + " -e " + str(0) + " -p " + str(p / 10) + " -v " + str(0.0)
#         subprocess.call(cmd, shell=True)
#         print(">>>>> ", p)

for i in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
    for j in range(10):
        cmd = cmd_base + " -s " + str(0) + " -e " + str(0) + " -p " + str(0.0) + " -v " + str(i)
        subprocess.call(cmd, shell=True)
        print(">>>>> ", i, j)