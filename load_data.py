import numpy as np
import random
import scipy
from scipy.sparse import csr_matrix
import gc

import torch
import torch_geometric as tg

import warnings
warnings.filterwarnings('ignore')

def load_IMDB(local_test=True, device='cpu', state_mode=1):
    # load relation data
    m_vs_d = np.load('../input/IMDB_processed/m_vs_d.npy')
    m_vs_d = scipy.sparse.csr_matrix(m_vs_d)
    m_vs_a = np.load('../input/IMDB_processed/m_vs_a.npy')
    m_vs_a = scipy.sparse.csr_matrix(m_vs_a)
    # load movie label data
    labels = np.load('../input/IMDB_processed/labels.npy')
    # OH-labels
    labels_oh = np.zeros((labels.size, labels.max()+1))
    labels_oh[np.arange(labels.size), labels] = 1
    # load features
    feature_m = scipy.sparse.load_npz('../input/IMDB_processed/feature_m.npz')
    feature_d = scipy.sparse.load_npz('../input/IMDB_processed/feature_d.npz')
    feature_a  = scipy.sparse.load_npz('../input/IMDB_processed/feature_a.npz')
    num_m = feature_m.shape[0]
    num_d = feature_d.shape[0]
    num_a = feature_a.shape[0]
    print('M: {}, D: {}, A: {}'.format(num_m, num_d, num_a))

    if local_test:
        selected_m = np.random.choice(np.arange(num_m), 500)
        selected_d = np.where(m_vs_d.toarray()[selected_m]>0)[1]
        selected_a = np.where(m_vs_a.toarray()[selected_m]>0)[1]
        m_vs_d = m_vs_d[selected_m, :][:, selected_d]
        m_vs_a = m_vs_a[selected_m, :][:, selected_a]
        labels = labels[selected_m]
        labels_oh = labels_oh[selected_m]
        feature_m = feature_m[selected_m]
        feature_d = feature_d[selected_d]
        feature_a = feature_a[selected_a]
        num_m = feature_m.shape[0]
        num_d = feature_d.shape[0]
        num_a = feature_a.shape[0]
        print('It is local test mode')
        print('Sampled M: {}, D: {}, A: {}'.format(num_m, num_d, num_a))
    
    # convert data to torch data
    feature_m = torch.FloatTensor(feature_m.toarray())
    feature_d = torch.FloatTensor(feature_d.toarray())
    feature_a = torch.FloatTensor(feature_a.toarray())
    labels = torch.LongTensor(labels)
    labels_oh = torch.LongTensor(labels_oh)

    adj_graph = np.zeros(((num_m+num_d+num_a), (num_m+num_d+num_a)), dtype=int)
    adj_graph[:num_m, num_m:(num_m+num_d)] = m_vs_d.toarray() # md
    adj_graph[:num_m, (num_m+num_d):(num_m+num_d+num_a)] = m_vs_a.toarray() # ma
    adj_graph[num_m:(num_m+num_d), :num_m] = m_vs_d.transpose().toarray() # dm
    adj_graph[(num_m+num_d):(num_m+num_d+num_a), :num_m] = m_vs_a.transpose().toarray() # am
    adj_graph = scipy.sparse.csr_matrix(adj_graph)

    # set up across levels data
    data = tg.data.Data(x=torch.cat((feature_m, feature_d, feature_a)),
                        edge_index=tg.utils.from_scipy_sparse_matrix(adj_graph)[0], 
                        edge_weight=tg.utils.from_scipy_sparse_matrix(adj_graph)[1].to(torch.float),
                        y=labels,
                        y_oh=labels_oh).to(device)
    data.x = data.x / data.x.sum(1, keepdim=True).clamp(min=1)
    data.states = data.x

    del m_vs_d, m_vs_a
    del feature_m, feature_d, feature_a
    del labels, labels_oh
    gc.collect()

    return data, adj_graph, num_m, num_d, num_a


def develop_graph_IMDB(data, num_m, num_d, num_a, device):
    print('\nCorrect ID ranges： \n M: {} to {}; D: {} to {}; A: {} to {}'.format(
        0, num_m - 1, num_m, num_m + num_d - 1, num_m + num_d, num_m + num_d + num_a - 1
    ))
    # set seperate edge_index for decision coversion
    edge_all = data.edge_index.data.cpu().numpy()
    edge_index_md = edge_all[:, (edge_all[0] < num_m) & (num_m <= edge_all[1]) & (edge_all[1] < num_m + num_d)]
    print('MD has {} edges, M: {} to {}, D: {} to {}'.format(edge_index_md.shape[1],
                                                             edge_index_md[0].min(), edge_index_md[0].max(),
                                                             edge_index_md[1].min(), edge_index_md[1].max()))
    edge_index_dm = edge_all[:, (num_m <= edge_all[0]) & (edge_all[0] < num_m + num_d) & (edge_all[1] < num_m)]
    print('DM has {} edges, D: {} to {}, M: {} to {}'.format(edge_index_dm.shape[1],
                                                             edge_index_dm[0].min(), edge_index_dm[0].max(),
                                                             edge_index_dm[1].min(), edge_index_dm[1].max()))
    edge_index_ma = edge_all[:,
                    (edge_all[0] < num_m) & (num_m + num_d <= edge_all[1]) & (edge_all[1] < num_m + num_d + num_a)]
    print('MA has {} edges, M: {} to {}, A: {} to {}'.format(edge_index_ma.shape[1],
                                                             edge_index_ma[0].min(), edge_index_ma[0].max(),
                                                             edge_index_ma[1].min(), edge_index_ma[1].max()))
    edge_index_am = edge_all[:,
                    (num_m + num_d <= edge_all[0]) & (edge_all[0] < num_m + num_d + num_a) & (edge_all[1] < num_m)]
    print('AM has {} edges, A: {} to {}, M: {} to {}'.format(edge_index_am.shape[1],
                                                             edge_index_am[0].min(), edge_index_am[0].max(),
                                                             edge_index_am[1].min(), edge_index_am[1].max()))
    data.edge_index_ma = torch.LongTensor(edge_index_ma).to(device)
    data.edge_index_am = torch.LongTensor(edge_index_am).to(device)
    data.edge_index_md = torch.LongTensor(edge_index_md).to(device)
    data.edge_index_dm = torch.LongTensor(edge_index_dm).to(device)

    return data


def load_DBLP(local_test=True, device='cpu', state_mode=1):
    # load relation data
    p_vs_a = np.load('../input/DBLP_processed/p_vs_a.npy')
    p_vs_a = scipy.sparse.csr_matrix(p_vs_a)
    p_vs_t = np.load('../input/DBLP_processed/p_vs_t.npy')
    p_vs_t = scipy.sparse.csr_matrix(p_vs_t)
    p_vs_c = np.load('../input/DBLP_processed/p_vs_c.npy')
    p_vs_c = scipy.sparse.csr_matrix(p_vs_c)
    # load movie label data
    labels = np.load('../input/DBLP_processed/labels.npy')
    # OH-labels
    labels_oh = np.zeros((labels.size, labels.max()+1))
    labels_oh[np.arange(labels.size), labels] = 1
    # load features
    feature_a = scipy.sparse.load_npz('../input/DBLP_processed/feature_a.npz')
    feature_p = scipy.sparse.load_npz('../input/DBLP_processed/feature_p.npz')
    feature_t = scipy.sparse.load_npz('../input/DBLP_processed/feature_t.npz')
    feature_c = scipy.sparse.load_npz('../input/DBLP_processed/feature_c.npz')
    feature_c = scipy.sparse.csr_matrix(feature_c)
    num_a = feature_a.shape[0]
    num_p = feature_p.shape[0]
    num_t = feature_t.shape[0]
    num_c = feature_c.shape[0]
    print('A: {}, P: {}, T: {}, C: {}'.format(num_a, num_p, num_t, num_c))
    a_vs_p = p_vs_a.transpose()
    a_vs_t = np.dot(a_vs_p, p_vs_t)
    a_vs_c = np.dot(a_vs_p, p_vs_c)

    if local_test:
        selected_a = np.random.choice(np.arange(num_a), 500)
        selected_p = np.unique(np.where(a_vs_p.toarray()[selected_a]>0)[1])
        selected_t = np.unique(np.where(a_vs_t.toarray()[selected_a]>0)[1])
        selected_c = np.unique(np.where(a_vs_c.toarray()[selected_a]>0)[1])
        a_vs_p = a_vs_p[selected_a, :][:, selected_p]
        p_vs_a = p_vs_a[selected_p, :][:, selected_a]
        p_vs_t = p_vs_t[selected_p, :][:, selected_t]
        p_vs_c = p_vs_c[selected_p, :][:, selected_c]
        labels = labels[selected_a]
        labels_oh = labels_oh[selected_a]
        feature_a = feature_a[selected_a]
        feature_p = feature_p[selected_p]
        feature_t = feature_t[selected_t]
        feature_c = feature_c[selected_c]
        num_a = feature_a.shape[0]
        num_p = feature_p.shape[0]
        num_t = feature_t.shape[0]
        num_c = feature_c.shape[0]
        print('Sampled A: {}, P: {}, T: {}, C: {}'.format(num_a, num_p, num_t, num_c))
    
    feature_a = np.concatenate([feature_a.toarray(), np.zeros((feature_a.shape[0], 4231-feature_a.shape[1]))], axis=1)
    feature_p = feature_p.toarray()
    feature_t = np.concatenate([feature_t.toarray(), np.zeros((feature_t.shape[0], 4231-feature_t.shape[1]))], axis=1)
    feature_c = np.concatenate([feature_c.toarray(), np.zeros((feature_c.shape[0], 4231-feature_c.shape[1]))], axis=1)
    feature_a = torch.FloatTensor(feature_a)
    feature_p = torch.FloatTensor(feature_p)
    feature_t = torch.FloatTensor(feature_t)
    feature_c = torch.FloatTensor(feature_c)
    labels = torch.LongTensor(labels)
    labels_oh = torch.LongTensor(labels_oh)

    adj_graph = np.zeros((num_a+num_p+num_t+num_c, num_a+num_p+num_t+num_c), dtype=int)
    adj_graph[:num_a, num_a:(num_a+num_p)] = a_vs_p.toarray() # ap
    adj_graph[num_a:(num_a+num_p), :num_a] = p_vs_a.toarray() # pa
    adj_graph[num_a:(num_a+num_p), (num_a+num_p):(num_a+num_p+num_t)] = p_vs_t.toarray() # pt
    adj_graph[num_a:(num_a+num_p), (num_a+num_p+num_t):] = p_vs_c.toarray() # pc
    adj_graph[(num_a+num_p):(num_a+num_p+num_t), num_a:(num_a+num_p)] = p_vs_t.transpose().toarray() # tp
    adj_graph[(num_a+num_p+num_t):, num_a:(num_a+num_p)] = p_vs_c.transpose().toarray() # cp
    adj_graph = scipy.sparse.csr_matrix(adj_graph)

    # set up across levels data
    data = tg.data.Data(x=torch.cat((feature_a, feature_p, feature_t, feature_c)),
                        edge_index=tg.utils.from_scipy_sparse_matrix(adj_graph)[0],
                        edge_weight=tg.utils.from_scipy_sparse_matrix(adj_graph)[1].to(torch.float),
                        y=labels,
                        y_oh=labels_oh).to(device)
    data.x = data.x / data.x.sum(1, keepdim=True).clamp(min=1)
    data.states = data.x

    return data, adj_graph, num_a, num_p, num_t, num_c


def develop_graph_DBLP(data, num_a, num_p, num_t, num_c, device):
    print('\nCorrect ID ranges： \n A: {} to {}; P: {} to {}; T: {} to {}; C: {} to {}'.format(
        0, num_a - 1, num_a, num_a + num_p - 1, num_a + num_p,
        num_a + num_p + num_t - 1, num_a + num_p + num_t, num_a + num_p + num_t + num_c - 1,
        num_a + num_p + num_t, num_a + num_p + num_t + num_c - 1
    ))
    # set seperate edge_index for decision coversion
    edge_all = data.edge_index.data.cpu().numpy()
    edge_index_ap = edge_all[:, (edge_all[0] < num_a) & (num_a <= edge_all[1]) & (edge_all[1] < num_a + num_p)]
    print('AP has {} edges, A: {} to {}, P: {} to {}'.format(edge_index_ap.shape[1],
                                                             edge_index_ap[0].min(), edge_index_ap[0].max(),
                                                             edge_index_ap[1].min(), edge_index_ap[1].max()))
    edge_index_pa = edge_all[:, (num_a <= edge_all[0]) & (edge_all[0] < num_a + num_p) & (edge_all[1] < num_a)]
    print('PA has {} edges, P: {} to {}, A: {} to {}'.format(edge_index_pa.shape[1],
                                                             edge_index_pa[0].min(), edge_index_pa[0].max(),
                                                             edge_index_pa[1].min(), edge_index_pa[1].max()))
    edge_index_pt = edge_all[:,
                    (num_a <= edge_all[0]) & (edge_all[0] < num_a + num_p) & (num_a + num_p <= edge_all[1]) & \
                    (edge_all[1] < num_a + num_p + num_t)]
    print('PT has {} edges, P: {} to {}, T: {} to {}'.format(edge_index_pt.shape[1],
                                                             edge_index_pt[0].min(), edge_index_pt[0].max(),
                                                             edge_index_pt[1].min(), edge_index_pt[1].max()))
    edge_index_pc = edge_all[:,
                    (num_a <= edge_all[0]) & (edge_all[0] < num_a + num_p) & (num_a + num_p + num_t <= edge_all[1])]
    print('PC has {} edges, P: {} to {}, C: {} to {}'.format(edge_index_pc.shape[1],
                                                             edge_index_pc[0].min(), edge_index_pc[0].max(),
                                                             edge_index_pc[1].min(), edge_index_pc[1].max()))
    edge_index_tp = edge_all[:, (num_a + num_p <= edge_all[0]) & (edge_all[0] < num_a + num_p + num_t) & \
                                (num_a <= edge_all[1]) & (edge_all[1] < num_a + num_p)]
    print('TP has {} edges, T: {} to {}, P: {} to {}'.format(edge_index_tp.shape[1],
                                                             edge_index_tp[0].min(), edge_index_tp[0].max(),
                                                             edge_index_tp[1].min(), edge_index_tp[1].max()))
    edge_index_cp = edge_all[:, (num_a + num_p + num_t <= edge_all[0]) & \
                                (num_a <= edge_all[1]) & (edge_all[1] < num_a + num_p)]
    print('CP has {} edges, C: {} to {}, P: {} to {}'.format(edge_index_cp.shape[1],
                                                             edge_index_cp[0].min(), edge_index_cp[0].max(),
                                                             edge_index_cp[1].min(), edge_index_cp[1].max()))

    data.edge_index_ap = torch.LongTensor(edge_index_ap).to(device)
    data.edge_index_pa = torch.LongTensor(edge_index_pa).to(device)
    data.edge_index_pt = torch.LongTensor(edge_index_pt).to(device)
    data.edge_index_tp = torch.LongTensor(edge_index_tp).to(device)
    data.edge_index_pc = torch.LongTensor(edge_index_pc).to(device)
    data.edge_index_cp = torch.LongTensor(edge_index_cp).to(device)

    return data


def data_split(data, num_target, num_train, local_test):
    if num_train == 0:
        if local_test:
            # semi-sup
            train_val_nodes = random.sample(list(range(num_target)), 100)
            train_nodes = train_val_nodes[:50]
            val_nodes = train_val_nodes[50:]
            test_nodes = list(set(range(num_target)) - set(train_val_nodes))
        else:
            # semi-sup
            train_val_nodes = random.sample(list(range(num_target)), 800)
            train_nodes = train_val_nodes[:400]
            val_nodes = train_val_nodes[400:]
            test_nodes = list(set(range(num_target)) - set(train_val_nodes))
    else:
        # set up train val and test
        shuffle = list(range(num_target))
        random.shuffle(shuffle)
        train_nodes = shuffle[: num_target * num_train // 100]
        val_nodes = shuffle[num_target * num_train // 100: 2 * num_target * num_train // 100]
        test_nodes = shuffle[2 * num_target * num_train // 100:]

    train_mask = np.array([False] * num_target)
    val_mask = np.array([False] * num_target)
    test_mask = np.array([False] * num_target)
    train_mask[train_nodes] = True
    val_mask[val_nodes] = True
    test_mask[test_nodes] = True

    data.train_mask = torch.Tensor(train_mask).bool()
    data.val_mask = torch.Tensor(val_mask).bool()
    data.test_mask = torch.Tensor(test_mask).bool()
    print('\nThere are {} target nodes, train: {}, valid: {}, test: {}'.format(
        num_target, sum(data.train_mask), sum(data.val_mask), sum(data.test_mask)
    ))

    return data


def load_data(data_name, num_train, local_test, state_mode=1, device='cpu'):
    if data_name == 'IMDB':
        data, adj_graph, num_m, num_d, num_a = load_IMDB(
            local_test=local_test,
            state_mode=state_mode,
            device=device,
        )
        data = develop_graph_IMDB(
            data=data, num_m=num_m, num_d=num_d, num_a=num_a, device=device
        )
        data = data_split(
            data=data, num_train=num_train, num_target=num_m, local_test=local_test,
        )
        # M: 0; D: 1; A: 2; stop: 3;
        data.num_diff_actions = 4
        # MD, DM, MA, AM
        data.num_relation = 4  # 2
        data.num_target = data.num_m = num_m
        data.num_d = num_d
        data.num_a = num_a
    elif data_name == 'DBLP':
        data, adj_graph, num_a, num_p, num_t, num_c = load_DBLP(
            local_test=local_test,
            state_mode=state_mode,
            device=device,
        )
        data = develop_graph_DBLP(
            data=data, num_a=num_a, num_p=num_p, num_t=num_t, num_c=num_c, device=device
        )
        data = data_split(
            data=data, num_train=num_train, num_target=num_a, local_test=local_test,
        )
        # A: 0; P: 1; T: 2; V: 3; stop: 4;
        data.num_diff_actions = 5
        # AP, PA, PT, TP, PV, VP
        data.num_relation = 6  # 3
        data.num_target = data.num_a = num_a
        data.num_p = num_p
        data.num_t = num_t
        data.num_c = num_c
    data.data_name = data_name

    return data, adj_graph
