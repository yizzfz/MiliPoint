# from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/dgcnn_segmentation.py
import torch
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool
from torch import nn

class DGCNN(torch.nn.Module):
    def __init__(self, k=30, aggr='max', info=None):
        super().__init__()
        self.num_classes = info['num_classes']
        self.conv = nn.ModuleList([])
        n = 3
        conv_layer=(32, 32, 32)
        dense_layer=(1024, 1024, 256, 128)

        for layer in conv_layer:
            edgeconv = DynamicEdgeConv(MLP([n * 2, layer, layer, layer]), k, aggr)
            self.conv.append(edgeconv)
            n = layer

        self.lin1 = MLP([sum(conv_layer), dense_layer[0]])

        if self.num_classes is None:    # keypoint dataset
            num_points = info['num_keypoints']
            self.num_points = num_points
            point_branches = {}
            for i in range(num_points):
                point_branches[f'branch_{i}'] = MLP([*dense_layer, 3], dropout=0.5, norm=None)
            self.output = torch.nn.ModuleDict(point_branches)
        else:                           # identification or action 
            self.output = MLP([*dense_layer, self.num_classes], dropout=0.5, norm=None)

    def forward(self, data):
        batchsize = data.shape[0]
        npoints = data.shape[1]
        x = data.reshape((batchsize * npoints, 3))
        batch = torch.arange(batchsize).repeat_interleave(npoints).to(x.device)
        xs = []
        for conv in self.conv:
            x = conv(x, batch)
            xs.append(x)
    
        x4 = self.lin1(torch.cat(xs, dim=1))
        x5 = global_max_pool(x4, batch)

        if self.num_classes is None:    # keypoint dataset
            y = []
            for i in range(self.num_points):
                y.append(self.output[f'branch_{i}'](x5))
            y = torch.stack(y, dim=1)
        else:                           # identification or action 
            y = self.output(x5)
        return y


