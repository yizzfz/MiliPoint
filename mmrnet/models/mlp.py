import torch
import torch_geometric
from torch import nn

class MLP(torch.nn.Module):
    def __init__(self, info=None, dense_layer=(64, 256, 256, 128)):
        super().__init__()
        self.num_classes = info['num_classes']
        self.feature = torch_geometric.nn.MLP([
            info['max_points'] * info['stacks'] *3, 
            *dense_layer
        ], act='relu', dropout=0.5, norm=None)

        if self.num_classes is None:    # keypoint dataset
            num_points = info['num_keypoints']
            self.num_points = num_points
            point_branches = {}
            for i in range(num_points):
                point_branches[f'branch_{i}'] = nn.Linear(dense_layer[-1], 3)
            self.output = torch.nn.ModuleDict(point_branches)
        else:                           # identification or action 
            self.output = nn.Linear(dense_layer[-1], self.num_classes)

    def forward(self, data):
        x, batch = data, None
        batch_size, n_input_points, input_feature = x.shape
        x = x.reshape((batch_size, n_input_points * input_feature))
        x = self.feature(x)
        
        if self.num_classes is None:    # keypoint dataset
            y = []
            for i in range(self.num_points):
                y.append(self.output[f'branch_{i}'](x))
            y = torch.stack(y, dim=1)
        else:                           # identification or action 
            y = self.output(x)
        return y


