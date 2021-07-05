import os
import torch
import argparse
from tqdm import tqdm
from ogb.lsc import PCQM4MEvaluator
from torch_geometric.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from pcqm4m_data import MyPCQM4MDataset
from gin_graph import GINGraphPooling

from torch.utils.tensorboard import SummaryWriter


def train(model, device, loader, optimizer, criterion_fn):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)
        pred = model(batch).view(-1,)
        optimizer.zero_grad()
        loss = criterion_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader)):
            batch = batch.to(device)
            pred = model(batch).view(-1,)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)["mae"]


def test(model, device, loader):
    model.eval()
    y_pred = []

    with torch.no_grad():
        for _, batch in enumerate(loader):
            batch = batch.to(device)
            pred = model(batch).view(-1,)
            y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)
    return y_pred

device=0
num_layers=5
graph_pooling='sum'
emb_dim256,
drop_ratio=0.
save_test='store_true'
batch_size=512
epochs=100
weight_decay=0.00001
early_stop=10
num_workers=4
dataset_root="dataset"

# automatic dataloading and splitting
dataset = MyPCQM4MDataset(root=dataset_root)
split_idx = dataset.get_idx_split()
train_data = dataset[split_idx['train']]
valid_data = dataset[split_idx['valid']]
test_data = dataset[split_idx['test']]
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# automatic evaluator. takes dataset name as input
evaluator = PCQM4MEvaluator()
criterion_fn = torch.nn.MSELoss()

model = GINGraphPooling(**nn_params).to(device)
num_params = sum(p.numel() for p in model.parameters())

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

not_improved = 0
best_valid_mae = 9999
for epoch in range(1, epochs + 1):
	train_mae = train(model, device, train_loader, optimizer, criterion_fn)

	valid_mae = eval(model, device, valid_loader, evaluator)

	print({'Train': train_mae, 'Validation': valid_mae})

	if valid_mae < best_valid_mae:
		best_valid_mae = valid_mae

	scheduler.step()
	print(f'Best validation MAE so far: {best_valid_mae}')



