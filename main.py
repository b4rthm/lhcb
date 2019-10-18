import warnings
import random

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, radius_graph, global_max_pool
from torch_geometric.utils import degree  # noqa
from torch_geometric.transforms import RandomRotate

from dataset import LHCbDataset
from sampler import ImbalancedDatasetSampler

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

dataset = LHCbDataset(root='LHC-data')
dataset.data.pos = dataset.data.pos / 50.0  # Rescale points.

num_neg_examples = (dataset.data.y == 0).sum().item()
num_pos_examples = (dataset.data.y == 1).sum().item()
pos_weight = torch.tensor(num_pos_examples / num_neg_examples)

# seperating positive and negative examples, with train and test data for each
num_train_examples_pos = int(0.8 * num_pos_examples)
num_train_examples_neg = int(0.8 * num_neg_examples)
num_test_examples = len(dataset) - num_train_examples_pos - num_train_examples_neg

# dataset is sorted with positive examples first, data = train_pos, test_pos, test_neg, train_neg
train_dataset_pos = dataset[:num_train_examples_pos]
test_dataset = dataset[num_train_examples_pos:(num_train_examples_pos + num_test_examples)]
train_dataset_neg = dataset[(num_train_examples_pos + num_test_examples):]

assert(len(train_dataset_pos) + len(train_dataset_neg) + len(test_dataset) == len(dataset))
train_dataset = train_dataset_pos.__add__(train_dataset_neg)

# to test positive and negative examples seperately
num_test_examples_pos = num_pos_examples - num_train_examples_pos
test_dataset_pos = test_dataset[:num_test_examples_pos]
test_dataset_neg = test_dataset[num_test_examples_pos:]

# train and test dataloader, iterator
# the batch will contain the same label for every example
batch_size = 48

train_loader = DataLoader(train_dataset, batch_size, drop_last=True, num_workers=6,
                          sampler=ImbalancedDatasetSampler(train_dataset))

#train_loader_pos = DataLoader(train_dataset_pos, batch_size, shuffle=False, drop_last=True, num_workers=6)
#train_iter_pos = iter(train_loader_pos)
#train_loader_neg = DataLoader(train_dataset_neg, batch_size, shuffle=False, drop_last=True, num_workers=6) 
#train_iter_neg = iter(train_loader_neg)

test_loader = DataLoader(test_dataset, batch_size, shuffle=False, drop_last=True, num_workers=6)
test_loader_pos = DataLoader(test_dataset_pos, batch_size, shuffle=False, drop_last=True, num_workers=6)
test_loader_neg = DataLoader(test_dataset_neg, batch_size, shuffle=False, drop_last=True, num_workers=6)


def MLP(arg1, arg2, arg3):
    return Seq(Lin(arg1, arg2), ReLU(), Lin(arg2, arg3))


def augmentate_data(data):
    r = random.randint(0,3)
    if r == 0:
        degree = random.randint(-180,180)
        data = RandomRotate(degree)(data)
    return data

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = PointConv(MLP(00 + 3, 64, 64))
        self.conv2 = PointConv(MLP(64 + 3, 128, 128))
        self.conv3 = PointConv(MLP(128 + 3, 128, 256))
        self.lin1 = Lin(64 + 128 + 256, 128)  # Jumping Knowledge.
        self.lin2 = Lin(128, 64)
        self.lin3 = Lin(64, 1)

    def forward(self, pos, batch):
        edge_index = radius_graph(pos, r=0.01, batch=batch)


        # print(degree(edge_index[0], num_nodes=pos.size(0)).mean())
        # print(degree(edge_index[1], num_nodes=pos.size(0)).mean())

        x1 = F.relu(self.conv1(None, pos, edge_index))
        x2 = F.relu(self.conv2(x1, pos, edge_index))
        x3 = F.relu(self.conv3(x2, pos, edge_index))
        x = torch.cat([x1, x2, x3], dim=-1)
        x = global_max_pool(x, batch, size=batch_size) #48?

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x.flatten()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)

# 0.001
# 0.01
# 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#class CustomDataset(torch.utils.data.Dataset):
#    def __init__(self, pos_dataset, neg_dataset):
#        self.pos_dataset = pos_dataset
#        self.neg_dataset = neg_dataset
#        self.length_pos = len(pos_dataset)

#    def __len__(self):
#        return 2 * self.length

#    def __getitem__(self, idx):
#        if idx < self.length:
            # sample neg
#        else:
            #sample pos

#        return data


def train(epoch):
    model.train()
    total_loss = 0

    i = 0
    dict = {'0':0, '1':0 , '=':0}
    for data in train_loader:
        data.to(device)

        # check which label dominates
        dict = update_dict(data, dict)

        optimizer.zero_grad()
        out = model(data.pos, data.batch)
        loss = F.binary_cross_entropy_with_logits(out, data.y.to(out.dtype), pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.y.size(0)

        i += 1
        if i > 0 and i % 100 == 0:
            print('Epoch: {:03d}, {:04d}/{:04d}'.format(epoch, i, len(train_loader)))
            #test(small_train_loader)
            #test(small_test_loader)
    print(dict)
    return total_loss / len(train_loader)


def update_dict(data,dict):
    if (data.y == 0).sum() > batch_size/2:
        dict['0'] += 1
    elif (data.y == 0).sum() < batch_size/2:
        dict['1'] += 1
    else:
        dict['='] += 1
    return dict

def test(loader):
    model.eval()

    logits, target = [], []
    for data in loader:
        data = data.to(device)
        if data.pos.size(0) == 0:
            continue
        target.append(data.y)
        with torch.no_grad():
            logits.append(torch.sigmoid(model(data.pos, data.batch)))
    logits = torch.cat(logits, dim=0).to('cpu')
    target = torch.cat(target, dim=0).to('cpu')

    print(logits)
    print(target)

    accs, f1s = [], []
    for t in range(1, 21):  # Try out different thresholds.
        pred = (logits > (t / 20)).to(torch.long)
        accs.append(pred.eq(target).sum().item() / len(loader.dataset))
        f1s.append(metrics.f1_score(target, pred))


    acc = torch.tensor(accs).max().item()
    f1 = torch.tensor(f1s).max().item()
    auc = metrics.roc_auc_score(target, logits)

    print('Acc: {:.4f}, F1: {:.4f}, AUC: {:.4f}'.format(acc, f1, auc))

    #pred = (logits > 0.5).to(torch.long)
    #acc = pred.eq(target).sum().item() / len(loader.dataset)
    #f1 = metrics.f1_score(target, pred)
    #auc = 0
    #print('Acc: {:.4f}, F1: {:.4f}'.format(acc, f1))

    return acc, f1, auc


for epoch in range(1, 101):
    loss = train(epoch)
    print('Loss: {:.5f}'.format(loss))

    print('--- BEGIN COMPLETE TEST RUN ---')
    test(test_loader)
#    print('--- TEST POSITIVE EXAMPLES ---')
#    test(test_loader_pos)
#    print('--- TEST NEGATIVE EXAMPLES ---')
#    test(test_loader_neg)
    print('--- END COMPLETE TEST RUN ----\n')
#   torch.save(model.state_dict(), 'model_{:03d}.pt'.format(epoch))
