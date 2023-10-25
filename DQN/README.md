# Instructions to execute

1. First, create the virtual environment and activate the environment.
```
virtualenv -p python3 myenv
source myenv/bin/activate
```

2. Then, we install all the required packages.
```
pip install -r requirements.txt
```

3. Register custom gym environment.
```
pip install -e gym-environments/
```

4. Now we are ready to train a DQN agent. To do this, we must execute the following command. Notice that inside the *train_DQN.py* there are different hyperparameters that you can configure to set the training for different topologies, to define the size of the GNN model, etc.
```
python train_DQN.py
```

5. Finally, we can evaluate our trained model on different topologies executing the command below. Notice that in the *evaluate_DQN.py* script you must modify the hyperparameters of the model to match the ones from the trained model.
```
python evaluate_DQN.py -d ./Logs/expsample_DQN_agentLogs.txt -s 0 -e 0 -p 0.0 -v 0.0
```

The parameters:

* `-s` : The size of the randomly generated topology. Only used in randomly generated topology.
* `-e` : The randomizer seed used in the environment.
* `-p` : Maximun PLR.
* `-v` : Standard Deviation Indicator ($\gamma$).

We provide a script (`test.py`) to execute multiple experiments.