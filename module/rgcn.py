import torch
import torch.nn as nn
import numpy as np
# import dgl
from torch_geometric.nn import RGCNConv

def make_graph(speakers:torch.Tensor, device):
    edge_index, edge_type = [], []
    length = speakers.size(1)
    batch_size = speakers.size(0)
    total_len = 0
    for j in range(batch_size):
        length_max = 0
        for i in range(length - 1):
            print(speakers[j, i], speakers[j, i+1])
            if (speakers[j, i] == speakers[j, i+1]):
                length_max = i + 2
                break
        if (length_max == 0):
            length_max = length
        print(length_max)
        for l in range(length_max):
            for k in range(length_max):
                edge_index.append(torch.tensor([l + total_len, k + total_len]))
                if (speakers[j, l] == speakers[j, k]):
                    edge_type.append(1)
                else:
                    edge_type.append(0)
        total_len += length
    
    edge_index = torch.stack(edge_index).transpose(1, 0).to(device)
    edge_type = torch.tensor(edge_type).to(device)

    return edge_index, edge_type

class RGCN(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(RGCN, self).__init__()
        self.gcn = RGCNConv(input_size, output_size, 2)
    
    def forward(self, node_features, edge_index, edge_type):
        return self.gcn(node_features, edge_index, edge_type)
    


if __name__ == "__main__":
    device = "cpu"
    batch = torch.randn((15, 5))
    speakers = torch.tensor([[0, 1, 0, 1, 1], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]])
    edge_index, edge_type = make_graph(speakers, device)
    conv = RGCN(5, 5)
    out_feature = conv(batch,edge_index, edge_type)
    # for i in range(out_feature.shape[0]):
    #     node_feature = out_feature[i]
    #     # print(node_feature.shape)
    #     # print(node_feature)
    #     print(emotion_linear(node_feature))
    #     # Tiến hành xử lý với node_feature tại đây

    # print(emotion_linear(out_feature))

    print(edge_index.shape)
    print(edge_index)
    print(edge_type.shape)
    print(edge_type)