import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy
import random
import torch
import time


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class Normalizer(object):
    ''' Normalizer class that tracks the running statistics for normlization
    '''

    def __init__(self):
        ''' Initialize a Normalizer instance.
        '''
        self.mean = None
        self.std = None
        self.state_memory = []
        self.max_size = 10000
        self.length = 0

    def normalize(self, s):
        ''' Normalize the state with the running mean and std.

        Args:
            s (numpy.array): the input state

        Returns:
            a (int):  normalized state
        '''
        if self.length == 0:
            return s
        return (s - self.mean) / (self.std + 1e-8)

    def append(self, s):
        ''' Append a new state and update the running statistics

        Args:
            s (numpy.array): the input state
        '''
        if len(self.state_memory) > self.max_size:
            self.state_memory.pop(0)
        self.state_memory.append(s)
        self.mean = np.mean(self.state_memory, axis=0)
        self.std = np.mean(self.state_memory, axis=0)
        self.length = len(self.state_memory)

class Memory(object):
    ''' Memory for saving transitions
    '''

    def __init__(self, memory_size, batch_size):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
        '''
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, done):
        ''' Save transition into memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            done (boolean): whether the episode is finished
        '''
        if len(self.memory) > self.memory_size:
            # self.memory.pop(0)
            self.memory = self.memory[-self.memory_size:]
            # self.memory = random.sample(self.memory, self.memory_size)
        transition = Transition(state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''
        samples = random.sample(self.memory, self.batch_size)
        return map(np.array, zip(*samples))

class DQNAgent(object):
    def __init__(self,
                scope,
                replay_memory_size=20000,
                # replay_memory_init_size=100,
                update_target_estimator_every=1000,
                discount_factor=0.99,
                epsilon_start=1.0,
                epsilon_end=0.1,
                epsilon_decay_steps=200,
                batch_size=32,
                action_num=None,
                step_num=None,
                state_shape=None,
                norm_sample=10000,
                # norm_step=1,
                mlp_layers=None,
                agent_mode=None,
                learning_rate=0.0005,
                device=None):

        '''
        Q-Learning algorithm for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
            scope (str): The name of the DQN agent
            # env (object): The Environment.
            replay_memory_size (int): Size of the replay memory 允许agent存储的最大记忆
            # replay_memory_init_size (int): Number of random experiences to sampel when initializing
              the reply memory.
            update_target_estimator_every (int): Copy parameters from the Q estimator to the
              target estimator every N steps 每间隔多少训练次数就赋值Q模拟器的参数给目标模拟器
            discount_factor (float): Gamma discount factor 目的是为了让最终的step得到的结果最优，中间的步骤进行打折
            epsilon_start (int): Chance to sample a random action when taking an action.
              Epsilon is decayed over time and this is the start value 随机概率开始的数值
            epsilon_end (int): The final minimum value of epsilon after decaying is done 随机概率结束的数值
            epsilon_decay_steps (int): Number of steps to decay epsilon over 随机概率参与多少步的预测
            batch_size (int): Size of batches to sample from the replay memory 训练agent的时候，从记忆中随机提取多少个样本用于训练
            # evaluate_every (int): Evaluate every N steps 没用到，我们的max_step替代了这个参数
            action_num (int): The number of the actions 有多少个不同的可选步骤
            state_space (list): The space of the state vector 每一个步骤背后的state的空间
            norm_sample (int): The number of the sample used form noramlize state 用前多少个状态样本用于训练归一器
            mlp_layers (list): The layer number and the dimension of each layer in MLP
            learning_rate (float): The learning rate of the DQN agent
            device (torch.device): whether to use the cpu or gpu
        '''
        self.scope = scope
        # self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.action_num = action_num
        self.step_num = step_num
        self.norm_sample = norm_sample # in terms of total_t 
        self.get_agent_action = False
        self.agent_mode = agent_mode
        self.ready_init = False # whether is ready to generate samples
        self.ready_train = False # whether is ready to train agent
        self.reset_gnn = False # need to reset the GNN model of env

        # Torch device
        self.device = device

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # The epsilon decay scheduler
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # Create estimators
        self.q_estimator = Estimator(action_num=action_num, learning_rate=learning_rate, state_shape=state_shape, \
            mlp_layers=mlp_layers, device=self.device)
        self.target_estimator = Estimator(action_num=action_num, learning_rate=learning_rate, state_shape=state_shape, \
            mlp_layers=mlp_layers, device=self.device)

        # Create normalizer
        self.normalizer = Normalizer()

        # Create replay memory
        self.memory = Memory(replay_memory_size, batch_size)

    def seed(self, random_seed):
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    def make_actions(self, env, states):
        if self.agent_mode == 0:
            # random
            best_actions = np.random.choice(np.arange(self.action_num), size=states.shape[0])
        elif self.agent_mode == 1:
            # predict by batch
            A = self.predict_batch(states)
            best_actions = np.random.choice(np.arange(len(A)), p=A, size=states.shape[0])
        elif self.agent_mode == 2:
            # predict by each item of batch
            As = self.predict_batch(states)
            best_actions = []
            for A in As:
                best_actions.append(np.random.choice(np.arange(len(A)), p=A, size=1)[0])
            best_actions = np.array(best_actions)
        else:
            print('Wrong agent mode!')
        print('Stat of all generate actions: ', np.unique(best_actions, return_counts=True))
        print('Stat of valid generate actions: ', np.unique(best_actions[env.batch_val_index], return_counts=True))
        return best_actions

    def learn(self, env, total_timesteps):
        best_val, best_test = 0, 0
        for timestep in range(total_timesteps):
            ls_val, ls_test = [], []
            print('Timestep: {}'.format(timestep))
            batch_states = env.reset2()
            test_embeddings, test_labels = [], []
            # memory_actions = []
            while env.i < len(env.node_indexes):
                print('batch {}-{} starts'.format(env.i, env.i+env.batch_size))
                print('there are {} data: {} train, {} labeled, {} val, {} test'.format(\
                    len(env.batch_index), len(env.batch_node_index), len(env.batch_train_index),\
                    len(env.batch_val_index), len(env.batch_test_index)))
                batch_current_states = batch_states
                all_generated_actions = []
                for current_step in range(self.step_num):
                    print('*'*50)
                    best_actions = self.make_actions(env, states=batch_current_states)
                    all_generated_actions.append(best_actions)
                    all_actions = np.vstack(all_generated_actions)
                    # get feedback from env
                    batch_memory_reward, batch_done, batch_memory_actions, val_acc, test_acc, emb_test, lab_test = env.step2(all_actions, current_step)
                    # get batch next states
                    batch_next_states = env.next_states[env.batch_index]
                    # get batch memory next states
                    batch_memory_next_states = env.next_states[env.batch_val_index]
                    # get batch memory current states
                    batch_memory_current_states = deepcopy(env.next_states)
                    batch_memory_current_states[env.batch_index] = batch_current_states
                    batch_memory_current_states = batch_memory_current_states[env.batch_val_index]
                    
                    trajectories = zip(batch_memory_current_states, batch_memory_actions,\
                        batch_memory_reward, batch_memory_next_states, batch_done)
                    if self.ready_init:
                        for each in trajectories:
                            self.feed(each)
                    batch_current_states = batch_next_states
                test_embeddings.append(emb_test)
                test_labels.append(lab_test)
                # record results
                ls_val.extend([val_acc]*len(env.batch_val_index))
                ls_test.extend([test_acc]*len(env.batch_test_index))
                self.ready_init = True
                # end of everything
                batch_states = env.update()
            print('Timestep {} Val acc: {:.4f}, Test acc: {:.4f}'.format(timestep, np.mean(ls_val), np.mean(ls_test)))
            if np.mean(ls_val) > best_val:
                best_val, best_test = np.mean(ls_val), np.mean(ls_test)
                best_test_emb, best_test_labels = test_embeddings, test_labels
        loss = self.train()
        if self.reset_gnn:
            env.reset_gnn()
            self.reset_gnn = False
            print('Reset GNN model environment.')
        return loss, best_val, best_test, np.concatenate(best_test_emb, axis=0), np.concatenate(best_test_labels, axis=0)
        
    def feed(self, ts):
        ''' Store data in to replay buffer and train the agent. There are two stages.
            In stage 1, populate the Normalizer to calculate mean and std.
            The transition is NOT stored in the memory
            In stage 2, the transition is stored to the memory.

        Args:
            ts (list): a list of 5 elements that represent the transition
        '''
        (state, action, reward, next_state, done) = tuple(ts)
        if self.total_t < self.norm_sample:
            self.feed_norm(state)
        else:
            self.feed_memory(state, action, reward, next_state, done)
        # self.feed_memory(state, action, reward, next_state, done)
        self.total_t += 1

    def eval_step(self, states):
        ''' Predict the action for evaluation purpose.

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        '''
        q_values = self.q_estimator.predict_nograd(self.normalizer.normalize(states))
        # q_values = self.q_estimator.predict_nograd(states)
        best_actions = np.argmax(q_values, axis=1)
        return best_actions
    
    def predict_batch(self, states):
        # epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        # I guess here should be train_t, because here we save many samples in one train
        epsilon = self.epsilons[min(self.train_t, self.epsilon_decay_steps-1)]
        # q_values = self.q_estimator.predict_nograd(states)
        q_values = self.q_estimator.predict_nograd(self.normalizer.normalize(states))
        best_action = np.argmax(q_values, axis=1)
        if self.agent_mode == 1:
            # predict by batch
            A = np.ones(self.action_num, dtype=float) * epsilon / self.action_num
            for a in best_action:
                A[a] += (1.0 - epsilon)
            A = A/A.sum()
        elif self.agent_mode == 2:
            # predict by each item of batch
            A = np.ones((states.shape[0], self.action_num), dtype=float) * epsilon / self.action_num
            for idx, a in enumerate(best_action):
                A[idx][a] += (1.0 - epsilon)
        else:
            print('Wrong agent mode!')
        return A
    
    def train(self):
        ''' Train the network

        Returns:
            loss (float): The loss of the current batch.
        '''
        if self.ready_train:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample()

            # Calculate best next actions using Q-network (Double DQN)
            # states are already normalized while saving, so we don't need normalizer here anymore
            q_values_next = self.q_estimator.predict_nograd(next_state_batch)
            best_actions = np.argmax(q_values_next, axis=1)

            # Evaluate best next actions using Target-network (Double DQN)
            # states are already normalized while saving, so we don't need normalizer here anymore
            q_values_next_target = self.target_estimator.predict_nograd(next_state_batch)
            target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]

            # Perform gradient descent update
            state_batch = np.array(state_batch)

            loss = self.q_estimator.update(state_batch, action_batch, target_batch)
            print('\rINFO - Agent {}, memory {}, rl-loss: {:.5f}\n'.format(self.scope, self.total_t, loss), end='')

            # Update the target estimator
            if self.train_t % self.update_target_estimator_every == 0:
                self.target_estimator = deepcopy(self.q_estimator)
                self.reset_gnn = True
                print("\nINFO - Copied model parameters to target network.")
            self.train_t += 1
        else:
            loss = -1
        
        return loss

    def feed_norm(self, state):
        ''' Feed state to normalizer to collect statistics

        Args:
            state (numpy.array): the state that will be feed into normalizer
        '''
        self.normalizer.append(state)

    def feed_memory(self, state, action, reward, next_state, done):
        ''' Feed transition to memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            done (boolean): whether the episode is finished
        '''
        self.memory.save(self.normalizer.normalize(state), action, reward, self.normalizer.normalize(next_state), done)
        # self.memory.save(state, action, reward, next_state, done)

class Estimator(object):
    '''
    Approximate clone of rlcard.agents.dqn_agent.Estimator that
    uses PyTorch instead of Tensorflow.  All methods input/output np.ndarray.

    Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    '''

    def __init__(self, action_num=2, learning_rate=0.001, state_shape=None, mlp_layers=None, device=None):
        ''' Initilalize an Estimator object.

        Args:
            action_num (int): the number output actions
            state_shape (list): the shape of the state space
            mlp_layers (list): size of outputs of mlp layers
            device (torch.device): whether to use cpu or gpu
        '''
        self.action_num = action_num
        self.learning_rate=learning_rate
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers
        self.device = device

        # set up Q model and place it in eval mode
        qnet = EstimatorNetwork(action_num, state_shape, mlp_layers)
        qnet = qnet.to(self.device)
        self.qnet = qnet
        self.qnet.eval()

        # initialize the weights using Xavier init
        for p in self.qnet.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        # set up loss function
        self.mse_loss = nn.MSELoss(reduction='mean')

        # set up optimizer
        self.optimizer =  torch.optim.Adam(self.qnet.parameters(), lr=self.learning_rate)

    def predict_nograd(self, s):
        ''' Predicts action values, but prediction is not included
            in the computation graph.  It is used to predict optimal next
            actions in the Double-DQN algorithm.

        Args:
          s (np.ndarray): (batch, state_len)

        Returns:
          np.ndarray of shape (batch_size, NUM_VALID_ACTIONS) containing the estimated
          action values.
        '''
        with torch.no_grad():
            s = torch.from_numpy(s).float().to(self.device)
            q_as = self.qnet(s).cpu().numpy()
        return q_as

    def update(self, s, a, y):
        ''' Updates the estimator towards the given targets.
            In this case y is the target-network estimated
            value of the Q-network optimal actions, which
            is labeled y in Algorithm 1 of Minh et al. (2015)

        Args:
          s (np.ndarray): (batch, state_shape) state representation
          a (np.ndarray): (batch,) integer sampled actions
          y (np.ndarray): (batch,) value of optimal actions according to Q-target

        Returns:
          The calculated loss on the batch.
        '''
        self.optimizer.zero_grad()

        self.qnet.train()

        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        # (batch, state_shape) -> (batch, action_num)
        q_as = self.qnet(s)

        # (batch, action_num) -> (batch, )
        Q = torch.gather(q_as, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)

        # update model
        batch_loss = self.mse_loss(Q, y)
        batch_loss.backward()
        self.optimizer.step()
        batch_loss = batch_loss.item()

        self.qnet.eval()

        return batch_loss

class EstimatorNetwork(nn.Module):
    ''' The function approximation network for Estimator
        It is just a series of tanh layers. All in/out are torch.tensor
    '''

    def __init__(self, action_num=2, state_shape=None, mlp_layers=None):
        ''' Initialize the Q network

        Args:
            action_num (int): number of legal actions
            state_shape (list): shape of state tensor
            mlp_layers (list): output size of each fc layer
        '''
        super(EstimatorNetwork, self).__init__()

        self.action_num = action_num
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers

        # build the Q network
        layer_dims = [np.prod(self.state_shape)] + self.mlp_layers
        fc = [nn.Flatten()]
        for i in range(len(layer_dims)-1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=True))
            # fc.append(nn.Tanh())
            fc.append(nn.ReLU())
        fc.append(nn.Linear(layer_dims[-1], self.action_num, bias=True))
        # init lin weights, add by ZZ
        for item in fc:
            if isinstance(item, nn.Linear):
                nn.init.xavier_normal_(item.weight, gain=1.414)
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, s):
        ''' Predict action values

        Args:
            s  (Tensor): (batch, state_shape)
        '''
        return self.fc_layers(s)
