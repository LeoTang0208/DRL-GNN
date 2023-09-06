import numpy as np
import gym
import os
import gym_environments
import networkx as nx
import random
import matplotlib.pyplot as plt
import argparse
import mpnn as gnn
from collections import deque
import tensorflow as tf

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ENV_NAME_AGENT = 'GraphEnv-v1'
ENV_NAME = 'GraphEnv-v1'

SEED = 9
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
tf.random.set_seed(1)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/
# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)

NUMBER_EPISODES = 50
# We assume that the number of samples is always larger than the number of demands any agent can ever allocate
NUM_SAMPLES_EPSD = 100

# Set evaluation topology
graph_topology = 0 # 0==NSFNET, 1==GEANT2, 2==Small Topology, 3==GBN
# NEW! 4 == Random Graph

listofDemands = [8, 32, 64]

hparams = {
    'l2': 0.1,
    'dropout_rate': 0.01,
    'link_state_dim': 20,
    'readout_units': 35,
    'learning_rate': 0.0001, # 0.0001
    'batch_size': 32,
    'T': 4, #4, 
    'num_demands': len(listofDemands)
}

class SAPAgent:
    # Shortest Available Path
    # Select the shortest available path among the K paths
    def __init__(self):
        self.K = 4 #4

    def act(self, env, state, demand, n1, n2):
        pathList = env.allPaths[str(n1) +':'+ str(n2)]
        path = 0
        allocated = 0 # Indicates 1 if we allocated the demand, 0 otherwise
        new_state = np.copy(state)
        while allocated==0 and path < len(pathList) and path<self.K:
            currentPath = pathList[path]
            can_allocate = 1 # Indicates 1 if we can allocate the demand, 0 otherwise
            i = 0
            j = 1

            # 1. Iterate over pairs of nodes and check if we can allocate the demand
            while j < len(currentPath):
                if new_state[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] - demand < 0:
                    can_allocate = 0
                    break
                i = i + 1
                j = j + 1

            if can_allocate==1:
                return path
            
            path = path + 1

        # If we can't allocate it we just do it in the first path
        if allocated==0:
            return 0

class LBAgent:
    # Load Balancing agent
    # Selects the path among the K paths with uniform probability
    def __init__(self):
        self.K = 4 #4

    def act(self, env, state, demand, n1, n2):
        pathList = env.allPaths[str(n1) +':'+  str(n2)]
        new_state = np.copy(state)

        free_capacity = 0
        id_last_free = -1 # Indicates the id of the last free path where we will allocate the demand
        path = 0
        # Check if there are at least 2 paths
        while free_capacity < 2 and path < len(pathList) and path<self.K:
            currentPath = pathList[path]
            can_allocate = 1  # Indicates 1 if we can allocate the demand, 0 otherwise
            i = 0
            j = 1

            # 1. Iterate over pairs of nodes and check if we can allocate the demand
            while j < len(currentPath):
                if new_state[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] - demand < 0:
                    can_allocate = 0
                    break
                i = i + 1
                j = j + 1

            if can_allocate == 1:
                free_capacity = free_capacity + 1
                id_last_free = path
            path = path + 1

        # If we can't allocate nowhere
        if free_capacity == 0:
            return 0

        # If there is just one path to allocate we allocate it there
        elif free_capacity == 1:
            return id_last_free
        else:
            allocated = 0 # Indicates 1 if we allocated the demand, 0 otherwise
            while allocated==0:
                # -1 to convert a <= max_action <= b to a <= max_action < b
                max_action = min(self.K,len(pathList))-1 
                action = random.randint(0, max_action)
 
                currentPath = pathList[action]
                can_allocate = 1 # Indicates 1 if we can allocate the demand, 0 otherwise
                i = 0
                j = 1

                # 1. Iterate over pairs of nodes and check if we can allocate the demand
                while j < len(currentPath):
                    if new_state[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] - demand < 0:
                        can_allocate = 0
                        break
                    i = i + 1
                    j = j + 1

                if can_allocate == 1:
                    return action

def cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes) - 1):
            cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes

class DQNAgent:
    def __init__(self, env_nsfnet):
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.writer = None
        self.K = 4 #4
        self.listQValues = None
        self.action = None
        self.capacity_feature = None
        self.bw_demand_feature = np.zeros((env_nsfnet.numEdges,len(env_nsfnet.listofDemands)))

        self.global_step = 0
        self.primary_network = gnn.myModel(hparams)
        self.primary_network.build()
        self.target_network = gnn.myModel(hparams)
        self.target_network.build()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=hparams['learning_rate'],momentum=0.9,nesterov=True)

    def act(self, env, state, demand, source, destination, flagEvaluation):
        """
        Given a demand stored in the environment it allocates the K=4 shortest paths on the current 'state'
        and predicts the q_values of the K=4 different new graph states by using the GNN model.
        Picks the state according to epsilon-greedy approach. The flag=TRUE indicates that we are testing
        the model and thus, it won't activate the drop layers.
        """
        # Set to True if we need to compute K=4 q-values and take the maxium
        takeMax_epsilon = False
        # List of graphs
        listGraphs = []
        # List of graph features that are used in the cummax() call
        list_k_features = list()
        # Initialize action
        action = 0

        # We get the K-paths between source-destination
        pathList = env.allPaths[str(source) +':'+ str(destination)]
        path = 0

        # 1. Implement epsilon-greedy to pick allocation
        # If flagEvaluation==TRUE we are EVALUATING => take always the action that the agent is saying has higher q-value
        # Otherwise, we are training with normal epsilon-greedy strategy
        if flagEvaluation:
            # If evaluation, compute K=4 q-values and take the maxium value
            takeMax_epsilon = True
        else:
            # If training, compute epsilon-greedy
            z = np.random.random()
            if z > self.epsilon:
                # Compute K=4 q-values and pick the one with highest value
                # In case of multiple same max values, return the first one
                takeMax_epsilon = True
            else:
                # Pick a random path and compute only one q-value
                path = np.random.randint(0, len(pathList))
                action = path

        # 2. Allocate (S,D, linkDemand) demand using the K shortest paths
        while path < len(pathList):
            state_copy = np.copy(state)
            currentPath = pathList[path]
            i = 0
            j = 1

            # 3. Iterate over paths' pairs of nodes and allocate demand to bw_allocated
            while (j < len(currentPath)):
                state_copy[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][1] = demand
                i = i + 1
                j = j + 1

            # 4. Add allocated graphs' features to the list. Later we will compute their q-values using cummax
            listGraphs.append(state_copy)
            features = self.get_graph_features(env, state_copy)
            list_k_features.append(features)

            if not takeMax_epsilon:
                # If we don't need to compute the K=4 q-values we exit
                break

            path = path + 1

        vs = [v for v in list_k_features]

        # We compute the graphs_ids to later perform the unsorted_segment_sum for each graph and obtain the 
        # link hidden states for each graph.
        graph_ids = [tf.fill([tf.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]
        first_offset = cummax(vs, lambda v: v['first'])
        second_offset = cummax(vs, lambda v: v['second'])

        tensors = ({
            'graph_id': tf.concat([v for v in graph_ids], axis=0),
            'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
            'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0),
            'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
            'num_edges': tf.math.add_n([v['num_edges'] for v in vs]),
            }
        )        

        # Predict qvalues for all graphs within tensors
        self.listQValues = self.primary_network(tensors['link_state'], tensors['graph_id'], tensors['first'],
                        tensors['second'], tensors['num_edges'], training=False).numpy()

        if takeMax_epsilon:
            # We take the path with highest q-value
            action = np.argmax(self.listQValues)
        else:
            return path, list_k_features[0]

        return action, list_k_features[action]
    
    def get_graph_features(self, env, copyGraph):
        """
        We iterate over the converted graph nodes and take the features. The capacity and bw allocated features
        are normalized on the fly.
        """
        self.bw_demand_feature.fill(0.0)
        # Normalize capacity feature
        # self.capacity_feature = (copyGraph[:,0] - 100.00000001) / 200.0
        cap = np.array(copyGraph[:, 0])
        max_cap = np.amax(cap)
        self.capacity_feature = (copyGraph[:, 0] - (max_cap / 2)) / max_cap

        itera = 0
        for i in copyGraph[:, 1]:
            if i == 8:
                self.bw_demand_feature[itera][0] = 1
            elif i == 32:
                self.bw_demand_feature[itera][1] = 1
            elif i == 64:
                self.bw_demand_feature[itera][2] = 1
            itera = itera + 1
        
        sample = {
            'num_edges': env.numEdges,  
            'length': env.firstTrueSize,
            'betweenness': tf.convert_to_tensor(value=env.between_feature, dtype=tf.float32),
            'bw_allocated': tf.convert_to_tensor(value=self.bw_demand_feature, dtype=tf.float32),
            'capacities': tf.convert_to_tensor(value=self.capacity_feature, dtype=tf.float32),
            'plr': tf.convert_to_tensor(value=env.plr_feature, dtype=tf.float32),
            'first': tf.convert_to_tensor(env.first, dtype=tf.int32),
            'second': tf.convert_to_tensor(env.second, dtype=tf.int32)
        }

        sample['capacities'] = tf.reshape(sample['capacities'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['betweenness'] = tf.reshape(sample['betweenness'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['plr'] = tf.reshape(sample['plr'][0:sample['num_edges']], [sample['num_edges'], 1])
        
        hiddenStates_ = tf.concat([sample['capacities'], sample['betweenness'], sample['bw_allocated']], axis=1)
        # print(">>>>>", tf.shape(hiddenStates_))

        hiddenStates = tf.concat([sample['capacities'], sample['betweenness'], sample['bw_allocated'], sample['plr']], axis=1) # NEW!
        # print("!!!!!", tf.shape(hiddenStates))

        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 3 - hparams['num_demands']]]) # PROBLEM? The params dont match, dont change now
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")
        # print("?????", tf.shape(link_state))

        inputs = {'link_state': link_state, 'first': sample['first'][0:sample['length']],
                  'second': sample['second'][0:sample['length']], 'num_edges': sample['num_edges']}
        
        """
        1.  link_state: 21*20 tensor, 21 --> num_edges, 
            20 --> capacity range [-0.5, 0.5], 
            betweenness range [-0.5, 0.5], 
            hidden_states [0/1, 0/1, 0/1] for [64, 32, 8], 
            15 for padding filled with 0
            
        2.  first, second: All adjacent links based on links in the graph
            >> 0 1      link no. 0
            >>>> 0 2    link no. 1
            >>>> 0 3    link no. 2
            >>>> 1 2    link no. 3
            >>>> 1 7    link no. 4
            
            translates to first, second:
            
            [0, 0, 0, 0]
            [1, 2, 3, 4]
            
        3. num_edges: num of edges in graph
        """

        return inputs

def exec_lb_model_episodes(experience_memory, graph_topology):
    
    env_lb = gym.make(ENV_NAME)
    env_lb.seed(SEED)
    env_lb.generate_environment(graph_topology, listofDemands, rand_size, rand_seed, plr_max, std_dev)
    
    link_u = []
    link_v = []
    for i, j in env_lb.ordered_edges:
        link_u.append(i)
        link_v.append(j)
    link_load = [0 for i in range(len(link_u))]
    # print(link_u)
    # print(link_v)

    agent = LBAgent()
    rewards_lb = np.zeros(NUMBER_EPISODES)
    demands_lb = np.zeros(NUMBER_EPISODES)

    rewardAdd = 0
    demandAdd = 0
    reward_it = 0
    iter_episode = 0 # Iterates over samples within the same episode
    new_episode = True
    wait_for_new_episode = False
    new_episode_it = 0 # Iterates over EPISODES
    while iter_episode < len(experience_memory):
        if new_episode:
            new_episode = False
            demand = experience_memory[iter_episode][1]
            source = experience_memory[iter_episode][2]
            destination = experience_memory[iter_episode][3]
            state = env_lb.eval_sap_reset(demand, source, destination)

            action = agent.act(env_lb, state, demand, source, destination)
            new_state, reward, done, _, _, _, currentpath, _ = env_lb.make_step(state, action, demand, source, destination)
            
            for i in range(len(currentpath) - 1):
                u = currentpath[i]
                v = currentpath[i + 1]
                for j in range(len(link_load)):
                    if (((u == link_u[j]) and (v == link_v[j])) or ((u == link_v[j]) and (v == link_u[j]))):
                        link_load[j] = link_load[j] + demand
            
            env_lb.demand = demand
            env_lb.source = source
            env_lb.destination = destination
            rewardAdd = rewardAdd + reward
            if (not done):
                demandAdd = demandAdd + demand
            state = new_state

            if done:
                rewards_lb[reward_it] = rewardAdd
                demands_lb[reward_it] = demandAdd
                reward_it = reward_it + 1
                wait_for_new_episode = True

            iter_episode = iter_episode + 1
        else:
            if experience_memory[iter_episode][0] != new_episode_it:
                print("LB ERROR! The experience replay buffer needs more samples/episode")
                os.kill(os.getpid(), 9)

            demand = experience_memory[iter_episode][1]
            source = experience_memory[iter_episode][2]
            destination = experience_memory[iter_episode][3]
            action = agent.act(env_lb, state, demand, source, destination)
            new_state, reward, done, _, _, _, currentpath, _ = env_lb.make_step(state, action, demand, source, destination)
            
            for i in range(len(currentpath) - 1):
                u = currentpath[i]
                v = currentpath[i + 1]
                for j in range(len(link_load)):
                    if (((u == link_u[j]) and (v == link_v[j])) or ((u == link_v[j]) and (v == link_u[j]))):
                        link_load[j] = link_load[j] + demand
            
            env_lb.demand = demand
            env_lb.source = source
            env_lb.destination = destination
            rewardAdd = rewardAdd + reward
            if (not done):
                demandAdd = demandAdd + demand
            state = new_state

            if done:
                rewards_lb[reward_it] = rewardAdd
                demands_lb[reward_it] = demandAdd
                reward_it = reward_it + 1
                wait_for_new_episode = True

            iter_episode = iter_episode + 1
        if wait_for_new_episode:
            rewardAdd = 0
            demandAdd = 0
            wait_for_new_episode = False
            new_episode = True
            new_episode_it = new_episode_it + 1
            iter_episode = new_episode_it*NUM_SAMPLES_EPSD
    
    return rewards_lb, link_load, demands_lb

def exec_sap_model_episodes(experience_memory, graph_topology):
    
    env_sap = gym.make(ENV_NAME)
    env_sap.seed(SEED)
    env_sap.generate_environment(graph_topology, listofDemands, rand_size, rand_seed, plr_max, std_dev)
    
    link_u = []
    link_v = []
    for i, j in env_sap.ordered_edges:
        link_u.append(i)
        link_v.append(j)
    link_load = [0 for i in range(len(link_u))]

    agent = SAPAgent()
    rewards_sap = np.zeros(NUMBER_EPISODES)
    demands_sap = np.zeros(NUMBER_EPISODES)

    rewardAdd = 0
    demandAdd = 0
    reward_it = 0
    iter_episode = 0  # Iterates over samples within the same episode
    new_episode = True
    wait_for_new_episode = False
    new_episode_it = 0  # Iterates over EPISODES
    while iter_episode < len(experience_memory):
        if new_episode:
            new_episode = False
            demand = experience_memory[iter_episode][1]
            source = experience_memory[iter_episode][2]
            destination = experience_memory[iter_episode][3]
            state = env_sap.eval_sap_reset(demand, source, destination)

            action = agent.act(env_sap, state, demand, source, destination)
            new_state, reward, done, _, _, _, currentpath, _ = env_sap.make_step(state, action, demand, source, destination)
            
            for i in range(len(currentpath) - 1):
                u = currentpath[i]
                v = currentpath[i + 1]
                for j in range(len(link_load)):
                    if (((u == link_u[j]) and (v == link_v[j])) or ((u == link_v[j]) and (v == link_u[j]))):
                        link_load[j] = link_load[j] + demand
            
            env_sap.demand = demand
            env_sap.source = source
            env_sap.destination = destination
            rewardAdd = rewardAdd + reward
            if (not done):
                demandAdd = demandAdd + demand
            state = new_state

            if done:
                rewards_sap[reward_it] = rewardAdd
                demands_sap[reward_it] = demandAdd
                reward_it = reward_it + 1
                wait_for_new_episode = True

            iter_episode = iter_episode + 1
        else:
            if experience_memory[iter_episode][0]!=new_episode_it:
                print("SAP ERROR! The experience replay buffer needs more samples/episode")
                os.kill(os.getpid(), 9)

            demand = experience_memory[iter_episode][1]
            source = experience_memory[iter_episode][2]
            destination = experience_memory[iter_episode][3]
            action = agent.act(env_sap, state, demand, source, destination)
            new_state, reward, done, _, _, _, currentpath, _ = env_sap.make_step(state, action, demand, source, destination)
            
            for i in range(len(currentpath) - 1):
                u = currentpath[i]
                v = currentpath[i + 1]
                for j in range(len(link_load)):
                    if (((u == link_u[j]) and (v == link_v[j])) or ((u == link_v[j]) and (v == link_u[j]))):
                        link_load[j] = link_load[j] + demand
            
            env_sap.demand = demand
            env_sap.source = source
            env_sap.destination = destination
            rewardAdd = rewardAdd + reward
            if (not done):
                demandAdd = demandAdd + demand
            state = new_state

            if done:
                rewards_sap[reward_it] = rewardAdd
                demands_sap[reward_it] = demandAdd
                reward_it = reward_it + 1
                wait_for_new_episode = True

            iter_episode = iter_episode + 1
        if wait_for_new_episode:
            rewardAdd = 0
            demandAdd = 0
            wait_for_new_episode = False
            new_episode = True
            new_episode_it = new_episode_it + 1
            iter_episode = new_episode_it * NUM_SAMPLES_EPSD
    
    return rewards_sap, link_load, demands_sap

def exec_dqn_model_episodes(experience_memory, env_dqn, agent):
    
    link_u = []
    link_v = []
    for i, j in env_dqn.ordered_edges:
        link_u.append(i)
        link_v.append(j)
    link_load = [0 for i in range(len(link_u))]
    
    rewards_dqn = np.zeros(NUMBER_EPISODES)
    demands_dqn = np.zeros(NUMBER_EPISODES)

    rewardAdd = 0
    demandAdd = 0
    reward_it = 0
    iter_episode = 0  # Iterates over samples within the same episode
    new_episode = True
    wait_for_new_episode = False
    new_episode_it = 0  # Iterates over EPISODES
    while iter_episode < len(experience_memory):
        if new_episode:
            new_episode = False
            demand = experience_memory[iter_episode][1]
            source = experience_memory[iter_episode][2]
            destination = experience_memory[iter_episode][3]
            state = env_dqn.eval_sap_reset(demand, source, destination)

            action, state_action = agent.act(env_dqn, state, demand, source, destination, True)
            new_state, reward, done, new_demand, new_source, new_destination, currentpath, _ = env_dqn.make_step(state, action, demand, source, destination)
            
            for i in range(len(currentpath) - 1):
                u = currentpath[i]
                v = currentpath[i + 1]
                for j in range(len(link_load)):
                    if (((u == link_u[j]) and (v == link_v[j])) or ((u == link_v[j]) and (v == link_u[j]))):
                        link_load[j] = link_load[j] + demand
            
            rewardAdd = rewardAdd + reward
            if (not done):
                demandAdd = demandAdd + demand
            state = new_state
            if done:
                rewards_dqn[reward_it] = rewardAdd
                demands_dqn[reward_it] = demandAdd
                reward_it = reward_it + 1
                wait_for_new_episode = True
            iter_episode = iter_episode + 1
        else:
            if experience_memory[iter_episode][0] != new_episode_it:
                print("DQNAgent ERROR! The experience replay buffer needs more samples/episode")
                os.kill(os.getpid(), 9)

            demand = experience_memory[iter_episode][1]
            source = experience_memory[iter_episode][2]
            destination = experience_memory[iter_episode][3]

            action, state_action = agent.act(env_dqn, state, demand, source, destination, True)
            new_state, reward, done, new_demand, new_source, new_destination, currentpath, _ = env_dqn.make_step(state, action, demand, source, destination)
            
            for i in range(len(currentpath) - 1):
                u = currentpath[i]
                v = currentpath[i + 1]
                for j in range(len(link_load)):
                    if (((u == link_u[j]) and (v == link_v[j])) or ((u == link_v[j]) and (v == link_u[j]))):
                        link_load[j] = link_load[j] + demand
            
            rewardAdd = rewardAdd + reward
            if (not done):
                demandAdd = demandAdd + demand
            state = new_state
            if done:
                rewards_dqn[reward_it] = rewardAdd
                demands_dqn[reward_it] = demandAdd
                reward_it = reward_it + 1
                wait_for_new_episode = True
            iter_episode = iter_episode + 1
        if wait_for_new_episode:
            rewardAdd = 0
            demandAdd = 0
            wait_for_new_episode = False
            new_episode = True
            new_episode_it = new_episode_it + 1
            if new_episode_it%5==0:
                print("DQN Episode >>> ", new_episode_it)
            iter_episode = new_episode_it * NUM_SAMPLES_EPSD
    
    return rewards_dqn, link_load, demands_dqn

if __name__ == "__main__":
    # python evaluate_DQN.py -d ./Logs/expsample_DQN_agentLogs.txt

    # Parse logs and get best model
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-d', help='data file', type=str, required=True, nargs='+')
    parser.add_argument('-s', type=int, required=True)
    parser.add_argument('-e', type=int, required=True)
    parser.add_argument('-p', type=float, required=True)
    parser.add_argument('-v', type=float, required=True)
    args = parser.parse_args()

    aux = args.d[0].split(".")
    aux = aux[1].split("exp")
    differentiation_str = str(aux[1].split("Logs")[0])

    if not os.path.exists("./Images"):
        os.makedirs("./Images")
    
    # TODO: Parse rand_size and rand_seed
    rand_size = args.s
    rand_seed = args.e
    plr_max = args.p
    std_dev = args.v
    # print(">>", rand_size, rand_seed)

    topo = ""
    if graph_topology==0:
        topo = "NSFNET"
    elif graph_topology==1:
        topo = "GEANT2"
    elif graph_topology==2:
        topo = "Small_Top"
    elif graph_topology==3:
        topo = "GBN"
    else:
        topo = "Rand"

    # Uncomment the following if you want to store the demands in a file
    # store_experiences = open("Traffic_demands_"+topo+"_1K.txt", "w")
    model_id = 0
    with open(args.d[0]) as fp:
        for line in reversed(list(fp)):
            arrayLine = line.split(":")
            if arrayLine[0]=='MAX REWD':
                model_id = int(arrayLine[2].split(",")[0])
                break

    env_dqn = gym.make(ENV_NAME_AGENT)
    env_dqn.seed(SEED)
    env_dqn.generate_environment(graph_topology, listofDemands, rand_size, rand_seed, plr_max, std_dev)
    
    dqn_agent = DQNAgent(env_dqn)
    checkpoint_dir = "./models" + "sample_DQN_agent_orig"
    
    # checkpoint_dir = "./Results" + "/models" + differentiation_str
    # model_id = 33
    
    checkpoint = tf.train.Checkpoint(model=dqn_agent.primary_network, optimizer=dqn_agent.optimizer)
    # Restore variables on creation if a checkpoint exists.
    checkpoint.restore(checkpoint_dir + "/" + str(hparams['T']) + "/ckpt-" + str(model_id))
    print("Load model " + checkpoint_dir + "/" + str(hparams['T']) + "/ckpt-" + str(model_id))

    iters = np.zeros(NUMBER_EPISODES)

    experience_memory = deque(maxlen=NUMBER_EPISODES*NUM_SAMPLES_EPSD)

    # Generate lists of determined size of demands. The different agents will iterate over the same list
    for ep_num in range(NUMBER_EPISODES):
        for sample in range(NUM_SAMPLES_EPSD):
            demand = np.random.choice(listofDemands)
            source = np.random.choice(env_dqn.nodes)

            # We pick a pair of SOURCE,DESTINATION different nodes
            while True:
                destination = np.random.choice(env_dqn.nodes)
                if destination != source:
                    # We generate unique demands that don't overlap with existing topology edges
                    experience_memory.append((ep_num, demand, source, destination))
                    #cstore_experiences.write(str(ep_num)+","+str(source)+","+str(destination)+","+str(demand)+"\n")
                    break

    # store_experiences.close()

    rewards_lb, load_lb, demands_lb = exec_lb_model_episodes(experience_memory, graph_topology)
    rewards_sap, load_sap, demands_sap = exec_sap_model_episodes(experience_memory, graph_topology)
    rewards_dqn, load_dqn, demands_dqn = exec_dqn_model_episodes(experience_memory, env_dqn, dqn_agent)
    
    link_u = []
    link_v = []
    for i, j in env_dqn.ordered_edges:
        link_u.append(i)
        link_v.append(j)
    
    cap = []
    for i, j in env_dqn.ordered_edges:
        # print(env_dqn.graph.get_edge_data(i, j)["capacity"])
        cap.append(env_dqn.graph.get_edge_data(i, j)["capacity"])
    cap = np.array(cap)
    
    # print(cap)
    # print(np.std(cap))
    
    print("!!!!!!!")
    # print(load_lb / NUMBER_EPISODES)
    # print([(_load / NUMBER_EPISODES) for _load in load_lb])
    
    lb_utilize = []
    sap_utilize = []
    dqn_utilize = []
    for i in range(len(cap)):
        lb_utilize.append((load_lb[i] / NUMBER_EPISODES) / cap[i])
        sap_utilize.append((load_sap[i] / NUMBER_EPISODES) / cap[i])
        dqn_utilize.append((load_dqn[i] / NUMBER_EPISODES) / cap[i])
    
    fig = plt.figure()
    pos = nx.spring_layout(env_dqn.graph, seed = 0)
    widths = 2 + 3 * (cap - cap.min()) / (cap.max() - cap.min())
    colors = np.array(dqn_utilize)
    nodes = nx.draw_networkx_nodes(env_dqn.graph, pos, node_color = "#A0CBE2")
    edges = nx.draw_networkx_edges(env_dqn.graph, pos, width = widths, edge_color = colors, edge_cmap = plt.cm.rainbow, edge_vmin = 0.0, edge_vmax = 1.0)
    labels = nx.draw_networkx_labels(env_dqn.graph, pos)
    plt.colorbar(edges)
    plt.title("Deep Reinforcement Learning")
    plt.axis("off")
    
    fig = plt.figure()
    pos = nx.spring_layout(env_dqn.graph, seed = 0)
    widths = 2 + 3 * (cap - cap.min()) / (cap.max() - cap.min())
    colors = np.array(sap_utilize)
    nodes = nx.draw_networkx_nodes(env_dqn.graph, pos, node_color = "#A0CBE2")
    edges = nx.draw_networkx_edges(env_dqn.graph, pos, width = widths, edge_color = colors, edge_cmap = plt.cm.rainbow, edge_vmin = 0.0, edge_vmax = 1.0)
    labels = nx.draw_networkx_labels(env_dqn.graph, pos)
    plt.colorbar(edges)
    plt.title("Shortest Available Path")
    plt.axis("off")
    
    fig = plt.figure()
    pos = nx.spring_layout(env_dqn.graph, seed = 0)
    widths = 2 + 3 * (cap - cap.min()) / (cap.max() - cap.min())
    colors = np.array(lb_utilize)
    nodes = nx.draw_networkx_nodes(env_dqn.graph, pos, node_color = "#A0CBE2")
    edges = nx.draw_networkx_edges(env_dqn.graph, pos, width = widths, edge_color = colors, edge_cmap = plt.cm.rainbow, edge_vmin = 0.0, edge_vmax = 1.0)
    labels = nx.draw_networkx_labels(env_dqn.graph, pos)
    plt.colorbar(edges)
    plt.title("Load Balancing")
    plt.axis("off")
    
    plt.show()
    
    print("!!!!!!!")

    #DQN
    mean_dqn = np.mean(rewards_dqn)
    fac_dqn = [(rewards_dqn[i] * np.max(listofDemands) / demands_dqn[i]) for i in range(len(rewards_dqn))]
    print(mean_dqn)

    #SAP
    mean_sap = np.mean(rewards_sap)
    fac_sap = [(rewards_sap[i] * np.max(listofDemands) / demands_sap[i]) for i in range(len(rewards_sap))]
    print(mean_sap)

    #LB
    mean_lb = np.mean(rewards_lb)
    fac_lb = [(rewards_lb[i] * np.max(listofDemands) / demands_lb[i]) for i in range(len(rewards_lb))]
    print(mean_lb)
    
    file = open("./result_logs/temp.txt", "a")
    file.write("----------------")
    
    reliability = np.mean([(env_dqn.betw_scale[i] / np.min(env_dqn.betw_scale) * env_dqn.plr_feature[i]) for i in range(env_dqn.numEdges)])
    for i in range(NUMBER_EPISODES):
        file.write(str(0) + " " + str(reliability) + " " + str(std_dev) + " " + str(fac_dqn[i]) + " " + str(fac_sap[i]) + " " + str(fac_lb[i]) + "\n")
        file.flush()
    
    file.write(str(1) + " " + str(reliability) + " " + str(std_dev) + " " + str(np.mean(fac_dqn)) + " " + str(np.mean(fac_sap)) + " " + str(np.mean(fac_lb)) + "\n")
    file.write(str(2) + " " + str(reliability) + " " + str(std_dev) + " " + str(np.mean(mean_dqn)) + " " + str(np.mean(mean_sap)) + " " + str(np.mean(mean_lb)) + "\n")
    file.flush()
    
    file.close()
