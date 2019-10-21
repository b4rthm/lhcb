import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.transforms import RandomRotate
from dataset import LHCbDataset
from sampler import ImbalancedDatasetSampler

dataset = LHCbDataset(root='LHC-data')
dataloader = DataLoader(dataset, 4, drop_last=True, num_workers=6, sampler=ImbalancedDatasetSampler(dataset))
_iterator = iter(dataloader)
data = next(_iterator)

print(data.pos[(data.y == 1)[data.batch]])

data_pos = Data(pos=data.pos[(data.y == 1)[data.batch]])
data_pos = RandomRotate(180, axis=0)(data_pos)
data.pos[(data.y == 1)[data.batch]] = data_pos.pos

print(data.pos[(data.y == 1)[data.batch]])
