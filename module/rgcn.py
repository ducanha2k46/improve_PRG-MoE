import torch
import torch.nn as nn
import numpy as np
import dgl
from torch_geometric.nn import RGCNConv

def make_graph(speakers:torch.Tensor, device):
    edge_index, edge_type = [], []
    length = speakers.size(1)
    batch_size = speakers.size(0)
    total_len = 0
    for j in range(batch_size):
        for l in range(length):
            for k in range(length):
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
    
    def forward(self, node_features,edge_index, edge_type):
        return self.gcn(node_features,edge_index ,edge_type)
    


if __name__ == "__main__":
    device = "cpu"
    batch = torch.randn((15, 5))
    speakers = torch.tensor([[0, 1, 1, 0, 0], [0, 1, 0, 1, 1], [0, 1, 0, 1, 0]])

    edge_index, edge_type = make_graph(speakers, device)

    conv = RGCN(5, 5)

    out_feature = conv(batch,edge_index, edge_type)

    print(out_feature.shape)




