import numpy as np
import pandas as pd
import scipy
from scipy import io as sio
from scipy.sparse import csr_matrix
from scipy.spatial import distance
import time
import random
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
from gym.spaces import Discrete
from gym import spaces
from copy import deepcopy
import sys
import gc
from eval_tools import evaluate_results_nc
from process_actions import filter_actions_IMDB, actions_to_agg_paths_IMDB
from process_actions import filter_actions_DBLP, actions_to_agg_paths_DBLP
from dqn_agent import DQNAgent

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as tg

class gnn_env_plus(object):
    def __init__(self, model, data, model_config, adj_graph, policy, device):
        super(gnn_env_plus, self).__init__()
        self.model_config = model_config
        self.init_model, self.data = model.to(device), data.to(device)
        self.train_indexes = torch.where(data.train_mask)[0].cpu().numpy()
        self.val_indexes = torch.where(data.val_mask)[0].cpu().numpy()
        self.test_indexes = torch.where(data.test_mask)[0].cpu().numpy()
        self.node_indexes = np.arange(data.num_target)
        self.all_node_indexes = np.arange(data.num_nodes)
        self.batch_size = model_config['batch_size']
        self.batch_train_size = int(len(self.train_indexes) / len(self.node_indexes) * self.batch_size)
        self.batch_val_size = int(len(self.val_indexes) / len(self.node_indexes) * self.batch_size)
        self.batch_test_size = self.batch_size-self.batch_train_size-self.batch_val_size
        self.i = 0
        self.val_acc = 0.0
        self.action_num = data.num_diff_actions
        obs = self.reset()
        self._set_observation_space(obs)
        self.policy = policy
        self.gnn_loss = torch.nn.NLLLoss()
        self.walk_length = model_config['walk_length']
        self.init_k_hop(adj_graph)
        self.device = device
        self.reward_mode = model_config['reward_mode']
        self.reward_coef = model_config['reward_coef']
        self.agent_action_mode = model_config['agent_action_mode']
        self.random_seed = model_config['SEED']
        self.stop_action = self.action_num-1
        self.data_name = data.data_name
        self.num_relation = data.num_relation
        if data.data_name == 'IMDB':
            self.num_target = data.num_target
            self.node_indexes = np.arange(data.num_target)
            self.all_node_indexes = np.arange(data.num_nodes)
            self.num_m = data.num_m
            self.num_d = data.num_d
            self.num_a = data.num_a
            self.edge_index_md = data.edge_index_md.cpu().numpy()
            self.edge_index_dm = data.edge_index_dm.cpu().numpy()
            self.edge_index_ma = data.edge_index_ma.cpu().numpy()
            self.edge_index_am = data.edge_index_am.cpu().numpy()
        elif data.data_name == 'DBLP':
            self.num_target = data.num_target
            self.node_indexes = np.arange(data.num_target)
            self.all_node_indexes = np.arange(data.num_nodes)
            self.num_a = data.num_a
            self.num_p = data.num_p
            self.num_t = data.num_t
            self.num_c = data.num_c
            self.edge_index_ap = data.edge_index_ap.cpu().numpy()
            self.edge_index_pa = data.edge_index_pa.cpu().numpy()
            self.edge_index_pt = data.edge_index_pt.cpu().numpy()
            self.edge_index_pc = data.edge_index_pc.cpu().numpy()
            self.edge_index_tp = data.edge_index_tp.cpu().numpy()
            self.edge_index_cp = data.edge_index_cp.cpu().numpy()

        # For Experiment #
        self.baseline_experience = self.model_config['baseline_experience']
        self.distance_func = distance.euclidean
        self.past_performance = [[] for _ in range(self.num_target)]
        self.past_p = [[] for _ in range(self.num_target)]
        self.past_dis_eu = [[] for _ in range(self.num_target)]
        # self.past_performance = [[[]] * self.walk_length for _ in range(num_m)]
        # self.past_p = [[[]] * self.walk_length for _ in range(num_m)]
        # self.past_dis_eu = [[[]] * self.walk_length for _ in range(num_m)]
        
    def seed(self, random_seed):
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        
    def init_k_hop(self, adj_graph):
        lamb = 1 * self.walk_length
        adj = adj_graph
        for _ in range(lamb-1):
            adj = np.dot(adj, adj_graph)
        self.adj = adj

    def reset(self):
        index = self.train_indexes[self.i]
        # self.reset_gnn()
        self.model = deepcopy(self.init_model)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.model_config['adam_lr'],
            weight_decay=self.model_config['weight_decay']
        )
        self.init_eval()
        state = self.init_embedding[index]
        return state

    def reset_gnn(self):
        # self.model = deepcopy(self.init_model)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), self.model_config['adam_lr'],
        #                                   weight_decay=self.model_config['weight_decay'])
        pass
        
    def _set_observation_space(self, observation):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
                
    def convert_agg_paths(self, paths):
        paths = paths[::-1]
        paths = [np.unique(item, axis=0) for item in paths] # merge duplicated paths
        paths = [torch.LongTensor(np.array(item)[np.where(np.array(item)[:, 1]>=0)].transpose()).\
                                     to(self.device) for item in paths] # drop -1 in paths
        paths = [p[[1, 0]] for p in paths]
        return paths
    
    def reset2(self):
        random.shuffle(self.node_indexes)
        random.shuffle(self.train_indexes)
        random.shuffle(self.val_indexes)
        random.shuffle(self.test_indexes)
        
        self.i, self.start_train, self.start_val, self.start_test = 0, 0, 0, 0
        end_train = min([self.start_train + self.batch_train_size, len(self.train_indexes)])
        end_val = min([self.start_val + self.batch_val_size, len(self.val_indexes)])
        end_test = min([self.start_test + self.batch_test_size, len(self.test_indexes)])
        self.batch_train_index = self.train_indexes[self.start_train:end_train]
        self.batch_val_index = self.val_indexes[self.start_val:end_val]
        self.batch_test_index = self.test_indexes[self.start_test:end_test]
        self.batch_node_index = np.concatenate(
            [self.batch_train_index, self.batch_val_index, self.batch_test_index]
        )
        
        if self.agent_action_mode == 1:
            self.batch_index = np.concatenate(
                [self.batch_train_index, self.batch_val_index, self.batch_test_index]
            )
        elif self.agent_action_mode == 2:
            self.batch_index = np.intersect1d(
                np.where(self.adj.toarray()[self.batch_node_index]>0)[1], self.all_node_indexes
            )
            self.batch_index = np.unique(
                np.concatenate([self.batch_node_index, self.batch_index], axis=0)
            )

        if self.agent_action_mode == 1:
            self.next_states = self.init_embedding[self.node_indexes]
        else:
            self.next_states = self.init_embedding[self.all_node_indexes]
        states = self.init_embedding[self.batch_index]
        return states
    
    def update(self):
        self.i += self.batch_size
        self.start_train += self.batch_train_size
        self.start_val += self.batch_val_size
        self.start_test += self.batch_test_size
        end_train = min([self.start_train + self.batch_train_size, len(self.train_indexes)])
        end_val = min([self.start_val + self.batch_val_size, len(self.val_indexes)])
        end_test = min([self.start_test + self.batch_test_size, len(self.test_indexes)])
        
        self.batch_train_index = self.train_indexes[self.start_train:end_train]
        self.batch_val_index = self.val_indexes[self.start_val:end_val]
        self.batch_test_index = self.test_indexes[self.start_test:end_test]
        self.batch_node_index = np.concatenate(
            [self.batch_train_index, self.batch_val_index, self.batch_test_index]
        )
        
        if self.agent_action_mode == 1:
            self.batch_index = np.concatenate(
                [self.batch_train_index, self.batch_val_index, self.batch_test_index]
            )
        elif self.agent_action_mode == 2:
            self.batch_index = np.intersect1d(
                np.where(self.adj.toarray()[self.batch_node_index] > 0)[1], self.all_node_indexes
            )
            self.batch_index = np.unique(np.concatenate([self.batch_node_index, self.batch_index], axis=0))

        self.init_eval()
        if self.agent_action_mode == 1:
            self.next_states = self.init_embedding[self.node_indexes]
        else:
            self.next_states = self.init_embedding[self.all_node_indexes]
        states = self.init_embedding[self.batch_index]
        return states
    
    def step_ahead(self, actions, current_step, node_index):
        # assign batch actions into target node range
        if self.agent_action_mode == 1:
            entire_actions = self.stop_action * np.ones((current_step+1, len(self.node_indexes)), dtype=int)
        # assign batch actions into all node range
        elif self.agent_action_mode == 2:
            entire_actions = self.stop_action * np.ones((current_step+1, len(self.all_node_indexes)), dtype=int)
        for idx in range(len(actions)):
            entire_actions[idx][self.batch_index] = actions[idx][self.batch_index]

        if self.data_name == 'IMDB':
            entire_actions = filter_actions_IMDB(
                actions=entire_actions, num_diff_actions=self.action_num,
                max_path_length=current_step + 1,
                num_m=self.num_m, num_d=self.num_d, num_a=self.num_a,
                mode=self.agent_action_mode
            )
        elif self.data_name == 'DBLP':
            entire_actions = filter_actions_DBLP(
                actions=entire_actions, num_diff_actions=self.action_num,
                max_path_length=current_step + 1,
                num_a=self.num_a, num_p=self.num_p, num_t=self.num_t, num_c=self.num_c,
                mode=self.agent_action_mode
            )
        memory_actions = entire_actions[-1]
        if self.data_name == 'IMDB':
            agg_paths, find_next_paths = actions_to_agg_paths_IMDB(
                entire_actions, 'both',
                self.edge_index_md, self.edge_index_dm,
                self.edge_index_ma, self.edge_index_am
            )
        elif self.data_name == 'DBLP':
            agg_paths, find_next_paths = actions_to_agg_paths_DBLP(
                entire_actions, 'both',
                self.edge_index_ap,
                self.edge_index_pa, self.edge_index_pt, self.edge_index_pc,
                self.edge_index_tp, self.edge_index_cp
            )
        agg_paths = self.convert_agg_paths(paths=agg_paths)

        return agg_paths, memory_actions, find_next_paths
    
    def get_batch_reward(self, N, val_acc, dis_eu_val, score_p_val):
        # record distance EU
        batch_dis_eu = np.zeros((len(self.node_indexes)))
        batch_dis_eu[self.batch_val_index] = dis_eu_val.flatten()
        # record distance P
        batch_p = np.zeros((len(self.node_indexes)))
        batch_p[self.batch_val_index] = score_p_val.flatten()
        # record acc
        batch_acc = np.zeros((len(self.node_indexes)))
        batch_acc[self.batch_val_index] = val_acc
        
        batch_reward = np.zeros((N))
        for index in self.batch_val_index:
            add, n_add = 0, 0
            if 'dis-eu' in self.reward_mode:
                if self.policy.ready_init:
                    # distance EU
                    add += self.reward_coef*(np.mean(self.past_dis_eu[index][-self.baseline_experience:]) - batch_dis_eu[index])
                    # add += self.reward_coef*(np.mean(self.past_dis_eu[index][current_step][-self.baseline_experience:]) - batch_dis_eu[index])
                    n_add += 1
                self.past_dis_eu[index].append(batch_dis_eu[index])
                # self.past_dis_eu[index][current_step].append(batch_dis_eu[index])
            if 'p' in self.reward_mode:
                if self.policy.ready_init:
                    # Possibility
                    add += self.reward_coef*(batch_p[index] - np.mean(self.past_p[index][-self.baseline_experience:]))
                    # add += self.reward_coef*(batch_p[index] - np.mean(self.past_p[index][current_step][-self.baseline_experience:]))
                    n_add += 1
                self.past_p[index].append(batch_p[index])
                # self.past_p[index][current_step].append(batch_p[index])
            if 'acc' in self.reward_mode:
                if self.policy.ready_init:
                    # ACC
                    add += self.reward_coef*(batch_acc[index] - np.mean(self.past_performance[index][-self.baseline_experience:]))
                    # add += self.reward_coef*(batch_acc[index] - np.mean(self.past_performance[index][current_step][-self.baseline_experience:]))
                    n_add += 1
                self.past_performance[index].append(batch_acc[index])
                # self.past_performance[index][current_step].append(batch_acc[index])
            # SUM UP
            if self.policy.ready_init:
                to_add = add/n_add
                batch_reward[index] += to_add
        return batch_reward

    def step2(self, actions, current_step, quiet=False):
        # actions to paths
        agg_paths, memory_actions, rnn_paths = self.step_ahead(
            actions, current_step, self.batch_index
        )
        for _ in range(self.walk_length-len(agg_paths)):
            agg_paths.append([])
        # perform train/val/test on paths
        loss_train, train_acc = self.train(agg_paths, rnn_paths)
        val_acc, dis_eu_val, score_p_val, test_acc, embedding_test, label_test = self.eval_batch(agg_paths, rnn_paths)
        # calculate batch reward scores
        batch_reward = self.get_batch_reward(memory_actions.shape[0], val_acc, dis_eu_val, score_p_val)
        
        batch_memory_reward = batch_reward[self.batch_val_index]
        batch_memory_actions = memory_actions[self.batch_val_index]
        batch_done = [current_step == (self.walk_length-1)] * len(self.batch_val_index)
        # batch_done = [True] * len(self.batch_val_index)
        
        print('Stat of all memory actions: ', np.unique(memory_actions, return_counts=True))
        print('Stat of valid memory actions: ', np.unique(batch_memory_actions, return_counts=True))            

        if not quiet:
            print('Step {}, Avg. reward {:.4f}, Train loss: {:.5f}, ACC: train {:.4f}, val {:.4f}, test {:.4f}'.format(
                current_step, np.mean(batch_reward[self.batch_val_index]),
                loss_train, train_acc, val_acc, test_acc
            ))
        
        return batch_memory_reward, batch_done, batch_memory_actions, val_acc, test_acc, embedding_test, label_test
        
    def init_eval(self):
        self.model.eval()
        _, _, embedding = self.model(self.data, [], [])
        
        embedding = embedding[: len(self.all_node_indexes)].data.cpu().numpy()
        
        # state_col = np.zeros((embedding.shape[0],), dtype=int)
        # state_oh = np.zeros((state_col.size, self.action_num))
        # state_oh[np.arange(state_col.size), state_col] = 1
        # self.state_col = state_oh
        # embedding = np.concatenate([embedding, self.state_col], axis=-1)
        
        self.init_embedding = embedding
            
    def train(self, agg_paths, rnn_paths):
        self.model.train()
        self.optimizer.zero_grad()
        
        pred, pred_dis, embedding = self.model(self.data, agg_paths, rnn_paths)
        # pred, pred_dis, embedding = self.model(self.data, [self.data.edge_index], rnn_paths)
        pred = pred[self.batch_train_index]
        # pred_dis = pred_dis.data.cpu().numpy()[self.batch_train_index]
        # embedding = embedding.data.cpu().numpy()[self.batch_train_index]
        y = self.data.y[self.batch_train_index]
        loss = self.gnn_loss(pred, y)
        # update
        loss.backward()
        self.optimizer.step()
        
        acc = accuracy_score(y_pred=np.argmax(pred.data.cpu().numpy(), axis=1), 
                             y_true=y.data.cpu().numpy())
        
        # self.next_states[self.batch_train_index] = np.concatenate([embedding,\
        #                                             self.state_col[self.batch_train_index]], axis=-1)
        
        return loss, acc
        
    def eval_batch(self, agg_paths, rnn_paths):
        self.model.eval()
        pred, pred_dis, embedding = self.model(self.data, agg_paths, rnn_paths, quiet=True)
        # pred, pred_dis, embedding = self.model(self.data, [self.data.edge_index], rnn_paths, quiet=True)
        # val part
        pred_val = pred.data.cpu().numpy()[self.batch_val_index]
        pred_dis_val = pred_dis.data.cpu().numpy()[self.batch_val_index]
        y_val = self.data.y.data.cpu().numpy()[self.batch_val_index]
        acc_val = accuracy_score(
            y_pred=np.argmax(pred_val, axis=1),
            y_true=y_val
        )
        score_p_val = np.array([p[q] for p, q in zip(pred_dis_val, y_val)])

        y_oh_val = self.data.y_oh.data.cpu().numpy()[self.batch_val_index]
        dis_eu_val = np.array([self.distance_func(p, q) for p, q in zip(pred_dis_val, y_oh_val)])
        
        # self.next_states[self.batch_val_index] = np.concatenate([embedding,\
        #                                             self.state_col[self.batch_val_index]], axis=-1)

        # test part
        pred_test = pred.data.cpu().numpy()[self.batch_test_index]
        embedding_test = embedding.data.cpu().numpy()[self.batch_test_index]
        y_test = self.data.y.data.cpu().numpy()[self.batch_test_index]
        acc_test = accuracy_score(
            y_pred=np.argmax(pred_test, axis=1),
            y_true=y_test
        )

        return acc_val, dis_eu_val, score_p_val, acc_test, embedding_test, y_test
