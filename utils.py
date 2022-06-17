import torch


def weights_init(m):
    if isinstance(m, torch.nn.Linear) or (isinstance(m, torch.nn.Parameter)):
        # nn.init.uniform_(m.weight.data)
        torch.nn.init.xavier_uniform_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
        # torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
    else:
        pass

