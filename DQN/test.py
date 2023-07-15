import subprocess
import random

random.seed()

cmd_base = "python evaluate_DQN.py -d ./Logs/expsample_DQN_agent_plr_4_Logs.txt"

# size = 40
for p in range(11):
    if (p != 0):
        for j in range(5):
            seed = random.randint(0, 50)
            cmd = cmd_base + " -s " + str(0) + " -e " + str(0) + " -p " + str(p/10)
            subprocess.call(cmd, shell=True)
            print(">>>>> ", p, j)
    else:
        seed = random.randint(0, 50)
        cmd = cmd_base + " -s " + str(0) + " -e " + str(0) + " -p " + str(p/10)
        subprocess.call(cmd, shell=True)
        print(">>>>> ", p)
