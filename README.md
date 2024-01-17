# Is Machine Learning the Best Option for Network Routing?
#### Liou Tang, Prashant Krishnamurthy, Mai Abdelhakim

---

Contact: <lit73@pitt.edu>

This is the code base for our paper *Is Machine Learning the Best Option for Network Routing?* accepted for IEEE International Conference on Communications (ICC) 2024.

Note that this repository is forked and modified from [here](https://github.com/knowledgedefinednetworking/DRL-GNN), the code base for the paper [P. Almasan et. al.](https://doi.org/10.1016/j.comcom.2022.09.029). **We have no affiliation with the author and creator of the aforementioned paper/code.**

## Instructions to execute

---

1. Install all the required packages.
```
pip install -r requirements.txt
```

2. Register custom gym environment.
```
pip install -e gym-environments/
```

3. To train the Deep Reinforcement Learning agent (DRL+GNN), execute the following command. Notice that inside the *train_DQN.py* there are different hyperparameters that you can configure to set the training for different topologies, to define the size of the GNN model, etc.
```
python train_DQN.py -s 0 -e 0 -p 0.0 -v 0.0
```

4. To evaluate the agent, execute the command below. Notice that in the *evaluate_DQN.py* script you must modify the hyperparameters of the model to match the ones from the trained model.
```
python evaluate_DQN.py -d ./Logs/expsample_DQN_agentLogs.txt -s 0 -e 0 -p 0.0 -v 0.0
```

The parameters:

* `-s` : The size of the randomly generated topology. Only used in randomly generated topology.
* `-e` : The randomizer seed used in the environment.
* `-p` : Maximum PLR.
* `-v` : Standard Deviation Indicator (denoted $\gamma$ in our paper).

We also provide a script (`test.py`) to help execute multiple experiments.

5. To train and/or evaluate the agent in different **topologies**, directly modify the `graph_topology` constant in `train_DQN.py` and/or `evaluate_DQN.py`. We have included some initial results regarding the agents' performances in different topologies but have not provided any detailed discussion.
