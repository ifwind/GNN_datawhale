import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, train_test_split_edges

dataset = 'Cora'
dataset = Planetoid('G:/chenyu/GNN/dataset/Cora', name='Cora', transform=T.NormalizeFeatures())#包括数据集的下载，若root路径存在数据集则直接加载数据集
data = dataset[0] #该数据集只有一个图len(dataset)：1，在这里才调用transform函数
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)
print(data)

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        return self.conv2(x, edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)  # [2,E]
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)  # *：element-wise乘法

    def decode_all(self, z):
        prob_adj = z @ z.t()  # @：矩阵乘法，自动执行适合的矩阵乘法函数
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def forward(self, x, pos_edge_index, neg_edge_index):
        return decode(encode(x, pos_edge_index), pos_edge_index, neg_edge_index)

def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train(data, model, optimizer, criterion):
    model.train()

    neg_edge_index = negative_sampling(  # 训练集负采样，每个epoch负采样样本可能不同
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))

    optimizer.zero_grad()
    # link_logits = model(data.x, data.train_pos_edge_index, neg_edge_index)
    z = model.encode(data.x, data.train_pos_edge_index)
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index).to(data.x.device)  # 训练集中正样本标签
    loss = criterion(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss

@torch.no_grad()
def mytest(data,model):
    model.eval()

    z = model.encode(data.x, data.train_pos_edge_index)

    results = []
    for prefix in ['val', 'test']:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()#计算链路存在的概率
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        results.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    return results

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features, 64).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = F.binary_cross_entropy_with_logits
best_val_auc = test_auc = 0
for epoch in range(1,101):
    loss=train(data,model,optimizer,criterion)
    val_auc,tmp_test_auc=mytest(data,model)
    if val_auc>best_val_auc:
        best_val_auc=val_auc
        test_auc=tmp_test_auc
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, Test: {test_auc:.4f}')
#预测
z=model.encode(data.x,data.train_pos_edge_index)
final_edge_index=model.decode_all(z)