import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from tqdm import tqdm
from torch_geometric.datasets import Reddit, Reddit2,PPI
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler
from torch_geometric.nn import SAGEConv
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True,lamb: float=1, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        self.lamb=lamb
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        adjmat,_ = add_self_loops(edge_index=edge_index,edge_weight=edge_weight)
        row, col = adjmat
        adjmat = SparseTensor(row=adjmat[0], col=adjmat[1], value=torch.ones(adjmat.shape[1]))
        deg_norm = degree(col, adjmat.size(0), dtype=torch.long).pow(-1)
        A_hat=adjmat.to_dense()
        A_hat=deg_norm*(A_hat)+self.lamb*torch.diag(A_hat)*torch.eye(A_hat.shape[0])
        A_hat=SparseTensor.from_dense(A_hat)

        x = x @ self.weight

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(A_hat, x=x)
        if self.bias is not None:
            out += self.bias
        return out

    def message_and_aggregate(self, A_hat: SparseTensor, x: Tensor) -> Tensor:
        return matmul(A_hat, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.convs = ModuleList(
            [GCNConv(in_channels, 128),
             GCNConv(128, out_channels)])

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(x, dim=-1)

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


def train():
    model.train()

    total_loss = total_nodes = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()

        nodes = batch.train_mask.sum().item()
        total_loss += loss.item() * nodes
        total_nodes += nodes

    return total_loss / total_nodes


@torch.no_grad()
def test():  # Inference should be performed on the full graph.
    model.eval()

    out = model.inference(data.x)
    y_pred = out.argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = y_pred[mask].eq(data.y[mask]).sum().item()
        accs.append(correct / mask.sum().item())
    return accs

if __name__ == '__main__':
    # 需要下载 https://data.dgl.ai/dataset/reddit.zip 到 data/Reddit 文件夹下
    # dataset = Reddit('data/Reddit')
    # dataset = PPI('data/PPI')
    dataset = Reddit2('data/Reddit2')
    data = dataset[0]

    # 图聚类
    cluster_data = ClusterData(data, num_parts=500, recursive=False,
                               save_dir=dataset.processed_dir)
    train_loader = ClusterLoader(cluster_data, batch_size=20, shuffle=True,
                                 num_workers=12)
    # 不聚类
    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1], batch_size=1024,
                                      shuffle=False, num_workers=12)

    device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset.num_features, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(1, 31):
        loss = train()
        if epoch % 5 == 0:
            train_acc, val_acc, test_acc = test()
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                  f'Val: {val_acc:.4f}, test: {test_acc:.4f}')
        else:
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
