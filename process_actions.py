import numpy as np
from copy import deepcopy


def filter_actions_IMDB(actions, num_diff_actions, max_path_length, num_m, num_d, num_a, mode):
    '''Filter illegal action combination
        mode 1: only make actions on Movie nodes in batch
        mode 2: make actions on all nodes in batch
    '''
    if mode == 1:
        added_actions = np.concatenate([np.zeros((1, num_m), dtype=int), actions])
    elif mode == 2:
        to_add = np.concatenate([np.zeros((1, num_m), dtype=int),
                                 np.ones((1, num_d), dtype=int),
                                 2 * np.ones((1, num_a), dtype=int)], axis=1)
        added_actions = np.concatenate([to_add, actions])
    
    # # case 1: stay at one node three steps
    # for i in range(max_path_length-1):
    #     illegal_index = np.where((added_actions[i]==added_actions[i+1]) & (added_actions[i]==added_actions[i+2]))[0]
    #     added_actions[i+2, illegal_index] = 3 * np.ones((1, illegal_index.shape[0]), dtype=int)
    # case 1: stay at one node two steps
    for i in range(max_path_length):
        illegal_index = np.where(added_actions[i] == added_actions[i+1])[0]
        added_actions[i+1, illegal_index] = (num_diff_actions-1) * np.ones((1, illegal_index.shape[0]), dtype=int)
    # # case 2: illegal paths in this dataset
    # for i in range(max_path_length):
    #     # 0-0
    #     illegal_index = np.where((added_actions[i]==0) & (added_actions[i+1]==0))[0]
    #     added_actions[i+1, illegal_index] = 3 * np.ones((1, illegal_index.shape[0]), dtype=int)
    #     # 1-1
    #     illegal_index = np.where((added_actions[i]==1) & (added_actions[i+1]==1))[0]
    #     added_actions[i+1, illegal_index] = 3 * np.ones((1, illegal_index.shape[0]), dtype=int)
    # case 1: stay at one node two steps
    for i in range(max_path_length):
        illegal_index = np.where(added_actions[i] == added_actions[i+1])[0]
        added_actions[i+1, illegal_index] = (num_diff_actions-1) * np.ones((1, illegal_index.shape[0]), dtype=int)
    # case 2: 1-2 / 2-1
    for i in range(max_path_length):
        illegal_index = np.where((added_actions[i] == 1) & (added_actions[i+1] == 2))[0]
        added_actions[i+1, illegal_index] = (num_diff_actions-1) * np.ones((1, illegal_index.shape[0]), dtype=int)
    for i in range(max_path_length):
        illegal_index = np.where((added_actions[i] == 2) & (added_actions[i+1] == 1))[0]
        added_actions[i+1, illegal_index] = (num_diff_actions-1) * np.ones((1, illegal_index.shape[0]), dtype=int)
    # case 3: Agent picks stop action
    for i in range(1, max_path_length+1):
        stop_index = np.where(added_actions[i] == num_diff_actions-1)[0]
        added_actions[i:, stop_index] = -1
    added_actions = np.where(added_actions == -1, (num_diff_actions-1), added_actions)
    
    return added_actions


def filter_actions_DBLP(actions, num_diff_actions, max_path_length, num_a, num_p, num_t, num_c, mode):
    '''Filter illegal action combination
        mode 1: only make actions on Movie nodes in batch
        mode 2: make actions on all nodes in batch
    '''
    if mode == 1:
        added_actions = np.concatenate([np.zeros((1, num_a), dtype=int), actions])
    elif mode == 2:
        to_add = np.concatenate([np.zeros((1, num_a), dtype=int), np.ones((1, num_p), dtype=int),
                                 2 * np.ones((1, num_t), dtype=int), 3 * np.ones((1, num_c), dtype=int)], axis=1)
        added_actions = np.concatenate([to_add, actions])

    # case 1: stay at one node two steps
    for i in range(max_path_length):
        illegal_index = np.where(added_actions[i] == added_actions[i + 1])[0]
        added_actions[i + 1, illegal_index] = (num_diff_actions - 1) * np.ones((1, illegal_index.shape[0]), dtype=int)
    # case 3: Agent picks stop action
    for i in range(1, max_path_length + 1):
        stop_index = np.where(added_actions[i] == num_diff_actions - 1)[0]
        added_actions[i:, stop_index] = -1
    added_actions = np.where(added_actions == -1, (num_diff_actions - 1), added_actions)

    return added_actions


def process_single_action_IMDB(action1, action2, col_id,
                          edge_index_md, edge_index_dm,
                          edge_index_ma, edge_index_am):
    '''
        m: 0; d: 1; a: 2;
        stop: 3;
    '''
    # across levels
    if action1 == 0 and action2 == 1:
        return edge_index_md[:, np.where((edge_index_md[0]==col_id))[0]].T
    elif action1 == 1 and action2 == 0:
        return edge_index_dm[:, np.where((edge_index_dm[0]==col_id))[0]].T   
    elif action1 == 0 and action2 == 2:
        return edge_index_ma[:, np.where((edge_index_ma[0]==col_id))[0]].T
    elif action1 == 2 and action2 == 0:
        return edge_index_am[:, np.where((edge_index_am[0]==col_id))[0]].T
    # stop
    elif action1 == 3 or action2 == 3:
        return np.array([[col_id, -1]])


# recursion
def process_actions_IMDB(action_col, col_id, result, edge_index_md, edge_index_dm,
                                                edge_index_ma, edge_index_am):
    # stop condition
    if len(action_col) < 2:
        return
    level_result = process_single_action_IMDB(
        action_col[0], action_col[1], col_id,
        edge_index_md, edge_index_dm, edge_index_ma, edge_index_am
    )
    
    for c in level_result:
        next_level_result = deepcopy(c.tolist())
        process_actions_IMDB(
            action_col[1:], c[1], next_level_result,
            edge_index_md, edge_index_dm, edge_index_ma, edge_index_am
        )
        result.append(next_level_result)


def process_single_action_DBLP(action1, action2, col_id, edge_index_ap,
                              edge_index_pa, edge_index_pt, edge_index_pc,
                              edge_index_tp, edge_index_cp):
    '''
        a: 0; p: 1; t: 2; c: 3
        stop: 4;
    '''
    # across levels
    if action1 == 0 and action2 == 1:
        return edge_index_ap[:, np.where((edge_index_ap[0] == col_id))[0]].T
    elif action1 == 1 and action2 == 0:
        return edge_index_pa[:, np.where((edge_index_pa[0] == col_id))[0]].T
    elif action1 == 1 and action2 == 2:
        return edge_index_pt[:, np.where((edge_index_pt[0] == col_id))[0]].T
    elif action1 == 1 and action2 == 3:
        return edge_index_pc[:, np.where((edge_index_pc[0] == col_id))[0]].T
    elif action1 == 2 and action2 == 1:
        return edge_index_tp[:, np.where((edge_index_tp[0] == col_id))[0]].T
    elif action1 == 3 and action2 == 1:
        return edge_index_cp[:, np.where((edge_index_cp[0] == col_id))[0]].T
    elif action1 == 0 and action2 == 2:
        return np.array([[col_id, -1]])
    elif action1 == 0 and action2 == 3:
        return np.array([[col_id, -1]])
    elif action1 == 2 and action2 == 0:
        return np.array([[col_id, -1]])
    elif action1 == 2 and action2 == 3:
        return np.array([[col_id, -1]])
    elif action1 == 3 and action2 == 0:
        return np.array([[col_id, -1]])
    elif action1 == 3 and action2 == 2:
        return np.array([[col_id, -1]])
    # stop
    elif action1 == 4 or action2 == 4:
        return np.array([[col_id, -1]])


# recursion
def process_actions_DBLP(action_col, col_id, result, edge_index_ap,
                              edge_index_pa, edge_index_pt, edge_index_pc,
                              edge_index_tp, edge_index_cp):
    # stop condition
    if len(action_col) < 2:
        return
    level_result = process_single_action_DBLP(action_col[0], action_col[1], col_id,
                                              edge_index_ap,
                                              edge_index_pa, edge_index_pt, edge_index_pc,
                                              edge_index_tp, edge_index_cp)

    for c in level_result:
        next_level_result = deepcopy(c.tolist())
        process_actions_DBLP(action_col[1:], c[1], next_level_result,
                             edge_index_ap,
                             edge_index_pa, edge_index_pt, edge_index_pc,
                             edge_index_tp, edge_index_cp)
        result.append(next_level_result)


def flat_list_A(arr, output, level_id):
    if type(arr[0]) == int:
        if len(output[0]) == 0:
            output[level_id] = [arr[:2]]
        else:
            output[level_id].append(arr[:2])
    if len(arr) == 2:
        return output
    else:
        level_id += 1
        for item in arr[2:]:
            flat_list_A(arr=item, output=output, level_id=level_id)


def generate_result_A(actions, result_final):
    result_A = [list()] * (actions.shape[0]-1)
    for k, v in result_final.items():
        for idx, c in enumerate(v):
            output = [[] for i in range(actions.shape[0]-1)]
            flat_list_A(c, output, level_id=0)
            if len(result_A[0]) == 0:
                result_A = output.copy()
            else:
                for step_id, item in enumerate(output):
                    result_A[step_id].extend(item)
    return result_A


def flat_list_B(arr, output, prefix):
    take_prefix = True
    for i in arr:
        if type(i) == int:
            prefix.append(i)
        if type(i) == list:
            if type(i[1]) == int:
                sublist = prefix.copy()
                take_prefix = False
                sublist.append(i[1])
            if len(i) > 2:
                flat_list_B(i[2:], output, sublist)  
            else:
                output.append(sublist)
    if take_prefix:
        output.append(prefix)
    return output


def generate_result_B(result_final):
    result_B = [list()] * len(result_final.keys())

    for k, v in result_final.items():
        line_output = list()
        for c in v:
            output = list()
            flat_list_B(c, output, list())
            line_output.extend(output)
        result_B[k] = line_output
    return result_B


def actions_to_agg_paths_IMDB(actions, mode, edge_index_md, edge_index_dm,
                                        edge_index_ma, edge_index_am):
    """
    mode:   A: obtain agg path, 2*n
            B: obatin find next arrival path, m*n
    """
    # TODO: parallise
    result_final = dict()
    N = actions.shape[1]
    for node_id in range(N):
        res = list()
        process_actions_IMDB(
            actions[:, node_id].tolist(), node_id, res,
            edge_index_md, edge_index_dm, edge_index_ma, edge_index_am
        )
        result_final[node_id] = res
    
    if mode == 'A':
        return generate_result_A(actions, result_final)
    elif mode == 'B':
        return generate_result_B(result_final)
    elif mode == 'both':
        return generate_result_A(actions, result_final), generate_result_B(result_final)


def actions_to_agg_paths_DBLP(actions, mode, edge_index_ap,
                              edge_index_pa, edge_index_pt, edge_index_pc,
                              edge_index_tp, edge_index_cp):
    # TODO: parallise
    result_final = dict()
    for node_id in range(actions.shape[1]):
        res = list()
        process_actions_DBLP(actions[:, node_id].tolist(), node_id, res,
                             edge_index_ap,
                             edge_index_pa, edge_index_pt, edge_index_pc,
                             edge_index_tp, edge_index_cp)
        result_final[node_id] = res

    if mode == 'A':
        return generate_result_A(actions, result_final)
    elif mode == 'B':
        return generate_result_B(result_final)
    elif mode == 'both':
        return generate_result_A(actions, result_final), generate_result_B(result_final)