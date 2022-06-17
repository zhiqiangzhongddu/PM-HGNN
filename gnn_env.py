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
from basic_GNN import HGNN

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as tg


class gnn_env(object):
    def __init__(self, model, data, model_config, adj_graph, policy, device):
        super(gnn_env, self).__init__()
        self.model, self.data = model.to(device), data.to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), model_config['adam_lr'],
            weight_decay=model_config['weight_decay']
        )
        self.train_indexes = torch.where(data.train_mask)[0].cpu().numpy()
        self.val_indexes = torch.where(data.val_mask)[0].cpu().numpy()
        self.test_indexes = torch.where(data.test_mask)[0].cpu().numpy()
        self.node_indexes = np.arange(data.num_target)
        self.all_node_indexes = np.arange(data.num_nodes)
        self.batch_size = model_config['batch_size']
        self.batch_train_size = int(len(self.train_indexes) / len(self.node_indexes) * self.batch_size)
        self.batch_val_size = int(len(self.val_indexes) / len(self.node_indexes) * self.batch_size)
        self.batch_test_size = int(len(self.test_indexes) / len(self.node_indexes) * self.batch_size)
        self.i = 0
        self.val_acc = 0.0
        self._set_action_space(data.num_diff_actions)
        obs = self.reset()
        self._set_observation_space(obs)
        self.policy = policy
        self.model_config = model_config
        # GNN
        self.max_layer = data.num_diff_actions
        self.gnn_loss = torch.nn.NLLLoss()
        # Agent
        self.walk_length = model_config['walk_length']
        self.init_k_hop(adj_graph)
        self.device = device
        self.reward_mode = model_config['reward_mode']
        self.reward_coef = model_config['reward_coef']
        self.early_stop = model_config['early_stop']
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
        self.baseline_experience = model_config['baseline_experience']
        self.distance_func = distance.euclidean
        self.past_performance = [[] for _ in range(self.num_target)]
        self.past_p = [[] for _ in range(self.num_target)]
        self.past_dis_eu = [[] for _ in range(self.num_target)]
        # self.past_performance = [[[]] * self.walk_length for _ in range(self.num_target)]
        # self.past_p = [[[]] * self.walk_length for _ in range(self.num_target)]
        # self.past_dis_eu = [[[]] * self.walk_length for _ in range(self.num_target)]
        
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
        state = self.data.states[index].cpu().numpy()
        return state

    def reset_gnn(self):
        pass

    def _set_action_space(self, _max):
        self.action_num = _max
        self.action_space = Discrete(_max) 

    def _set_observation_space(self, observation):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    
    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear) or (isinstance(m, torch.nn.Parameter)):
            # nn.init.uniform_(m.weight.data)
            torch.nn.init.xavier_uniform_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
            # torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')

    def stochastic_next_state(self, paths, indexes):
        find_next_paths = [np.array(item) for item in paths]
        for node_id in indexes:
            paths = find_next_paths[node_id]
            if sum(paths[:, -1]<0) < paths.shape[0]:
                paths = np.hstack((paths, -1 * np.ones((paths.shape[0], 1), dtype=int)))
            next_ids = paths[:, np.where(paths < 0)[1].min()-1]
            if (next_ids[0] != -1) & (next_ids[0] != node_id):
                # should not be divided by 2, make sure self-self distinguishable
                self.next_states[node_id] = (self.next_states[node_id] +
                                             self.data.states[next_ids].cpu().numpy().mean(0)) / 2  # or /2
                # self.next_states[node_id] = self.next_states[node_id] +\
                #                             self.data.states[next_ids].data.cpu().numpy().sum(0)
            else:
                self.next_states[node_id] = np.zeros((1, self.next_states.shape[1]))

    def convert_agg_paths(self, paths):
        paths = paths[::-1]
        paths = [np.unique(item, axis=0) for item in paths]  # merge duplicated paths
        paths = [torch.LongTensor(
            np.array(item)[np.where(np.array(item)[:, 1] >= 0)].transpose()
        ).to(self.device) for item in paths]  # drop -1 in paths
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
        self.batch_node_index = np.concatenate([self.batch_train_index, self.batch_val_index, self.batch_test_index])
        
        if self.agent_action_mode == 1:
            self.batch_index = np.concatenate([self.batch_train_index, self.batch_val_index, self.batch_test_index])
        elif self.agent_action_mode == 2:
            self.batch_index = np.intersect1d(
                np.where(self.adj.toarray()[self.batch_node_index] > 0)[1], self.all_node_indexes
            )
            self.batch_index = np.unique(np.concatenate([self.batch_node_index, self.batch_index], axis=0))

        if self.agent_action_mode == 1:
            self.next_states = self.data.states[self.node_indexes].cpu().numpy()
        else:
            self.next_states = self.data.states[self.all_node_indexes].cpu().numpy()
        states = self.data.states[self.batch_index].cpu().numpy()
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
        self.batch_node_index = np.concatenate([self.batch_train_index, self.batch_val_index, self.batch_test_index])
        if self.agent_action_mode == 1:
            self.batch_index = np.concatenate([self.batch_train_index, self.batch_val_index, self.batch_test_index])
        elif self.agent_action_mode == 2:
            self.batch_index = np.intersect1d(
                np.where(self.adj.toarray()[self.batch_node_index] > 0)[1], self.all_node_indexes
            )
            self.batch_index = np.unique(np.concatenate([self.batch_node_index, self.batch_index], axis=0))

        if self.agent_action_mode == 1:
            self.next_states = self.data.states[self.node_indexes].cpu().numpy()
        else:
            self.next_states = self.data.states[self.all_node_indexes].cpu().numpy()
        states = self.data.states[self.batch_index].cpu().numpy()
        return states
    
    def step_ahead(self, actions, current_step, node_index, update_state):
        # assign batch actions into all node range
        if self.agent_action_mode == 1:
            entire_actions = self.stop_action * np.ones((current_step+1, len(self.node_indexes)), dtype=int)
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

        if update_state:           
            memory_actions = None
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
            self.stochastic_next_state(paths=find_next_paths, indexes=node_index)
        else:
            memory_actions = entire_actions[-1]
            agg_paths = None
            find_next_paths = None
            
        return agg_paths, entire_actions, memory_actions, find_next_paths
    
    def step2(self, actions, current_step, quiet=False, test_mode=False):
        # go ahead to find the next states and agg paths
        actions_train = actions_val = actions_test = (self.stop_action)*np.ones_like(actions)
        replace_train = np.where(np.isin(self.batch_index, self.batch_train_index))[0]
        actions_train[:, replace_train] = actions[:, replace_train]
        replace_val = np.where(np.isin(self.batch_index, self.batch_val_index))[0]
        actions_val[:, replace_val] = actions[:, replace_val]
        replace_test = np.where(np.isin(self.batch_index, self.batch_test_index))[0]
        actions_test[:, replace_test] = actions[:, replace_test]
        # generate paths for diff types nodes
        _, entire_actions, memory_actions, _ = self.step_ahead(actions, current_step, self.batch_index, False)
        agg_paths_train, _, _, rnn_paths_train = self.step_ahead(actions_train, current_step,\
                                                                 self.batch_train_index, True)
        agg_paths_val, _, _, rnn_paths_val = self.step_ahead(actions_val, current_step,\
                                                             self.batch_val_index, True)
        agg_paths_test, _, _, rnn_paths_test = self.step_ahead(actions_test, current_step,\
                                                               self.batch_test_index, True)
        
        if not test_mode:
            self.batch_model = HGNN(
                pretrain_gnn=self.model_config['pretrain_gnn'],
                walk_length=self.walk_length,
                num_relations=self.num_relation,
                feat_dim=self.data.x.shape[1],
                hid_dim=self.model_config['hid_dim'],
                out_dim=self.data.y.unique().shape[0],
                gnn_type=self.model_config['gnn_type'],
                gnn_layers=self.model_config['gnn_layers'],
                agg_type=self.model_config['agg_type'],
                rnn_type=self.model_config['rnn_type'],
                dropout=self.model_config['dropout'],
                act_type=self.model_config['act_type'],
                device=self.device
            ).to(self.device)
            self.batch_model.apply(self.weights_init)
            self.batch_optimizer = torch.optim.Adam(
                self.batch_model.parameters(), self.model_config['adam_lr'],
                weight_decay=self.model_config['weight_decay']
            )
            best_val, best_test, best_i = 0, 0, 0
            best_test_emb, best_test_label = [], []
            for i in range(10001):
                loss_train, train_acc = self.train(agg_paths_train, rnn_paths_train)
                val_acc, dis_eu_val, score_p_val = self.eval_batch(agg_paths_val, rnn_paths_val)
                test_acc, embedding_test, label_test = self.test_batch(agg_paths_test, rnn_paths_test)
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = test_acc
                    best_dis_eu_val = dis_eu_val
                    best_score_p_val = score_p_val
                    best_i = i
                    best_test_emb = embedding_test
                    best_test_label = label_test
                else:
                    if i - best_i > self.early_stop:
                        break
            val_acc, test_acc, dis_eu_val, score_p_val = best_val, best_test, best_dis_eu_val, best_score_p_val
            embedding_test, label_test = best_test_emb, best_test_label
        else:
            self.batch_model, self.batch_optimizer = None, None
            loss_train, train_acc = self.train(agg_paths_train, rnn_paths_train, test_mode=True)
            val_acc, dis_eu_val, score_p_val = self.eval_batch(agg_paths_val, rnn_paths_val, test_mode=True)
            test_acc, embedding_test, label_test = self.test_batch(agg_paths_test, rnn_paths_test, test_mode=True)
            
        # record distance EU
        batch_dis_eu = np.zeros((len(self.node_indexes)))
        batch_dis_eu[self.batch_val_index] = dis_eu_val.flatten()
        # record distance P
        batch_p = np.zeros((len(self.node_indexes)))
        batch_p[self.batch_val_index] = score_p_val.flatten()
        # record acc
        batch_acc = np.zeros((len(self.node_indexes)))
        batch_acc[self.batch_val_index] = val_acc
        
        batch_reward = np.zeros((entire_actions.shape[1]))
        self.memory_index = self.batch_val_index
        for index in self.memory_index:
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
                # self.past_p[index].append(batch_p[index])
                self.past_p[index][current_step].append(batch_p[index])
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
        
        batch_memory_reward = batch_reward[self.memory_index]
        batch_memory_actions = memory_actions[self.memory_index]
        batch_done = [True] * len(self.memory_index)

        if not quiet:
            print('Step {}, Avg. reward {:.4f}, Train loss: {:.5f}, ACC: train {:.4f}, val {:.4f}, test {:.4f}'.format(
                current_step, np.mean(batch_reward[self.batch_val_index]),
                loss_train, train_acc, val_acc, test_acc
            ))
        return batch_memory_reward, batch_done, batch_memory_actions, val_acc, test_acc, embedding_test, label_test
    
    def train(self, agg_paths, rnn_paths, test_mode=False):
        if not test_mode:
            self.batch_model.train()
            self.batch_optimizer.zero_grad()
            pred, pred_dis, embedding = self.batch_model(self.data, agg_paths, rnn_paths)
            pred = pred[self.batch_train_index]
            pred_dis = pred_dis.data.cpu().numpy()[self.batch_train_index]
            embedding = embedding.data.cpu().numpy()[self.batch_train_index]
            y = self.data.y[self.batch_train_index]
            loss = self.gnn_loss(pred, y)
            # update
            loss.backward()
            self.batch_optimizer.step()
        else:
            self.model.train()
            self.optimizer.zero_grad()
            pred, pred_dis, embedding = self.model(self.data, agg_paths, rnn_paths)
            pred = pred[self.batch_train_index]
            pred_dis = pred_dis.data.cpu().numpy()[self.batch_train_index]
            embedding = embedding.data.cpu().numpy()[self.batch_train_index]
            y = self.data.y[self.batch_train_index]
            loss = self.gnn_loss(pred, y)
            # update
            loss.backward()
            self.optimizer.step()
            
        acc = accuracy_score(y_pred=np.argmax(pred.data.cpu().numpy(), axis=1), 
                             y_true=y.data.cpu().numpy())
        
        return loss, acc 
        
    def eval_batch(self, agg_paths, rnn_paths, test_mode=False):
        if not test_mode:
            self.batch_model.eval()
            pred, pred_dis, embedding = self.batch_model(self.data, agg_paths, rnn_paths)
        else:
            self.model.eval()
            pred, pred_dis, embedding = self.model(self.data, agg_paths, rnn_paths)
        pred = pred.data.cpu().numpy()[self.batch_val_index]
        pred_dis = pred_dis.data.cpu().numpy()[self.batch_val_index]
        # embedding = embedding.data.cpu().numpy()[self.batch_val_index]
        y = self.data.y.data.cpu().numpy()[self.batch_val_index]
        acc = accuracy_score(y_pred=np.argmax(pred, axis=1), 
                             y_true=y)
        score_p = np.array([p[q] for p, q in zip(pred_dis, y)])

        y_oh = self.data.y_oh.data.cpu().numpy()[self.batch_val_index]
        dis_eu = np.array([self.distance_func(p, q) for p, q in zip(pred_dis, y_oh)])
        
        return acc, dis_eu, score_p

    def test_batch(self, agg_paths, rnn_paths, test_mode=False):
        if not test_mode:
            self.batch_model.eval()
            pred, pred_dis, embedding = self.batch_model(self.data, agg_paths, rnn_paths)
        else:
            self.model.eval()
            pred, pred_dis, embedding = self.model(self.data, agg_paths, rnn_paths)
        pred = pred.data.cpu().numpy()[self.batch_test_index]
        pred_dis = pred_dis.data.cpu().numpy()[self.batch_test_index]
        embedding = embedding.data.cpu().numpy()[self.batch_test_index]
        y = self.data.y.data.cpu().numpy()[self.batch_test_index]
        acc = accuracy_score(y_pred=np.argmax(pred, axis=1), 
                             y_true=y)
        
        return acc, embedding, y
