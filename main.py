# sampling entfernt im small_train_loader
# Confusion Matrix
# batchnorm in MLP (before or after relu?)



# pos mit reinpacken, erste MLP auf 6 ändern und pos reinnehmen
# vor global max pool ein MLP, concat pos mit x
# mehrere z zusammenfassen, tracks in beide richtungen + selfloops
# z normalisieren


import time
import warnings
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.nn import DataParallel, Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import PointConv, radius_graph, global_max_pool
from torch_geometric.utils import degree  # noqa
from torch_geometric.transforms import RandomRotate

from dataset import LHCbDataset
from sampler import ImbalancedDatasetSampler

s_time = time.time()

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

dataset = LHCbDataset(root='LHC-data')

# dataset.data.pos = dataset.data.pos / 50.0  # Rescale points.

dataset_neg = Subset(dataset, range(0, 93485))
dataset_pos = Subset(dataset, range(93485, 99999))

num_neg_examples = len(dataset_neg)
num_pos_examples = len(dataset_pos)

train_dataset_neg = Subset(dataset_neg, range(int(0.8 * num_neg_examples)))
train_dataset_pos = Subset(dataset_pos, range(int(0.8 * num_pos_examples)))

small_train_dataset_neg = Subset(train_dataset_neg, range(6514))

test_dataset_neg = Subset(dataset_neg, range(int(0.8 * num_neg_examples), num_neg_examples))
test_dataset_pos = Subset(dataset_pos, range(int(0.8 * num_pos_examples), num_pos_examples))

# concatenating datasets
train_dataset = train_dataset_neg.__add__(train_dataset_pos)
small_train_dataset = small_train_dataset_neg.__add__(train_dataset_pos)
test_dataset = test_dataset_neg.__add__(test_dataset_pos)

batch_size = 32
num_workers = 6
radius = 0.7  # 0.7
lr = 0.001
augment = False

# Remove later
print_degree = True


# DataLoader
train_with_small = True
if train_with_small: print('Training with small_train_loader')

if train_with_small:
    train_loader = DataLoader(small_train_dataset, batch_size, drop_last=True, num_workers=num_workers)
else:
    train_loader = DataLoader(train_dataset, batch_size, drop_last=True, num_workers=num_workers,
                              sampler=ImbalancedDatasetSampler(train_dataset))
test_loader = DataLoader(test_dataset, batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

print('DataLoader Ready, Time Elapsed {:.0f}min'.format((time.time() - s_time)/60))


def MLP(arg1, arg2, arg3):
    # return Seq(Lin(arg1, arg2), ReLU(), Lin(arg2, arg3))
    return Seq(Lin(arg1, arg2), ReLU(), BN(arg2), Lin(arg2, arg3))


def augment_pos(data):
    data_pos = Data(pos=data.pos[(data.y == 1)[data.batch]])
    r = random.randint(0,3)
    if r == 0:
        degree = random.randint(-180,180)
        data_pos = RandomRotate(degree, axis=0)(data_pos)
        data.pos[(data.y == 1)[data.batch]] = data_pos.pos
    return data

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = PointConv(MLP(00 + 3, 64, 64))
        self.conv1_2 = PointConv(MLP(64 + 3, 64, 64))

        self.conv2 = PointConv(MLP(64 + 3, 128, 128))
        self.conv2_2 = PointConv(MLP(128 + 3, 128, 128))

        self.conv3 = PointConv(MLP(128 + 3, 256, 256))
        self.conv3_2 = PointConv(MLP(256 + 3, 256, 256))

        self.lin1 = Lin(2*64 + 2*128 + 2*256, 128)  # Jumping Knowledge.
        self.bn1 = BN(128)
        self.lin2 = Lin(128, 64)
        self.bn2 = BN(64)
        self.lin3 = Lin(64, 1)


    def forward(self, pos, batch, edge_index_tracks, edge_index_z):
        x1 = F.relu(self.conv1(None, pos, edge_index_tracks))
        x1_2 = F.relu(self.conv1_2(x1, pos, edge_index_z))

        x2 = F.relu(self.conv2(x1_2, pos, edge_index_tracks))
        x2_2 = F.relu(self.conv2_2(x2, pos, edge_index_z))

        x3 = F.relu(self.conv3(x2_2, pos, edge_index_tracks))
        x3_2 = F.relu(self.conv3_2(x3, pos, edge_index_z))

        x = torch.cat([x1, x1_2, x2, x2_2, x3, x3_2], dim=-1)
        x = global_max_pool(x, batch, size=batch_size)

        x = self.bn1(F.relu(self.lin1(x)))
        x = self.bn2(F.relu(self.lin2(x)))
        x = self.lin3(x)
        return x.flatten()


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
#model = DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train(epoch, loader):
    model.train()
    total_loss = 0

    i = 0
    dict = {'0':0, '1':0 , '=':0}
    for data in loader:
        data.to(device)
        if augment:
            data = augment_pos(data)

        # check which label dominates
        dict = update_dict(data, dict)

        optimizer.zero_grad()
        out = model(data.pos, data.batch, data.edge_index_tracks, data.edge_index_z)
        loss = F.binary_cross_entropy_with_logits(out, data.y.to(out.dtype))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.y.size(0)

        i += 1
        if i > 0 and i % 500 == 0:
            print('Epoch: {:03d}, {:04d}/{:04d}'.format(epoch, i, len(loader)))
    # print(dict)
    return total_loss / len(loader.dataset)


def update_dict(data,dict):
    if (data.y == 0).sum() > batch_size/2:
        dict['0'] += 1
    elif (data.y == 0).sum() < batch_size/2:
        dict['1'] += 1
    else:
        dict['='] += 1
    return dict


def recall(label, prediction, target_class):
    tp = torch.sum(prediction[label == target_class] == target_class).float()
    fn = torch.sum(prediction[label == target_class] != target_class).float()
    return (tp/(tp+fn)).item()

def precision(label, prediction, target_class):
    tp = torch.sum(prediction[label == target_class] == target_class).float()
    fp = torch.sum(prediction[label != target_class] == target_class).float()
    return (tp/(tp+fp)).item()

def f1_score(recall, precision):
    r_p = torch.tensor([recall, precision])
    if torch.any(torch.isnan(r_p)) or torch.all(r_p == 0):
        return 0
    return 2*precision*recall/(precision+recall)

def test(loader):
    model.eval()

    logits, target = [], []
    for data in loader:
        data = data.to(device)
        target.append(data.y)
        with torch.no_grad():
            logits.append(torch.sigmoid(model(data.pos, data.batch, data.edge_index_tracks, data.edge_index_z)))
    logits = torch.cat(logits, dim=0).to('cpu')
    target = torch.cat(target, dim=0).to('cpu')

    logits_0 = logits[target == 0]
    logits_1 = logits[target == 1]
    print('Target 0 Logit Mean: {:.4f}  Min: {:.4f}  Max: {:.4f}  Median: {:.4f}'.format(\
        logits_0.mean().item(), logits_0.min().item(), logits_0.max().item(), logits_0.median().item()))
    print('Target 1 Logit Mean: {:.4f}  Min: {:.4f}  Max: {:.4f}  Median: {:.4f}\n'.format(\
        logits_1.mean().item(), logits_1.min().item(), logits_1.max().item(), logits_1.median().item()))

    accs, recalls_0, precs_0, f1s_0 = [], [], [], []
    recalls_1, precs_1, f1s_1 = [], [], []
    confusion_matrixs = [] # contains (#target0_pred0, #target0_pred1, #target1_pred0, #target1_pred1)

    for t in range(1, 21):  # Try out different thresholds.
        pred = (logits > (t / 20)).to(torch.long)
        accs.append(pred.eq(target).sum().item() / len(loader.dataset))

#        recalls.append(metrics.recall_score(target, pred))
#        precs.append(metrics.precision_score(target, pred))
#        f1s.append(metrics.f1_score(target, pred))

        r = recall(target, pred, 0)
        p = precision(target, pred, 0)
        recalls_0.append(r)
        precs_0.append(p)
        f1s_0.append(f1_score(r,p))

        r = recall(target, pred, 1)
        p = precision(target, pred, 1)
        recalls_1.append(r)
        precs_1.append(p)
        f1s_1.append(f1_score(r,p))

        cm = (torch.sum(pred[target == 0] == 0).item(),
              torch.sum(pred[target == 0] == 1).item(),
              torch.sum(pred[target == 1] == 0).item(),
              torch.sum(pred[target == 1] == 1).item())
        confusion_matrixs.append(cm)

    auc = metrics.roc_auc_score(target, logits)

    f1s_0 = torch.tensor(f1s_0)
    f1s_1 = torch.tensor(f1s_1)

    for j in range(2):
        if j == 0:
            i = f1s_0.argmax()
        else:
            i = f1s_1.argmax()

        f1_0 = f1s_0[i].item()
        f1_1 = f1s_1[i].item()

        acc = torch.tensor(accs)[i].item()
        recall_0 = torch.tensor(recalls_0)[i].item()
        prec_0 = torch.tensor(precs_0)[i]
        if torch.isnan(prec_0):
            prec_0 = 0
        else:
            prec_0 = prec_0.item()

        recall_1 = torch.tensor(recalls_1)[i].item()
        prec_1 = torch.tensor(precs_1)[i]
        if torch.isnan(prec_1):
            prec_1 = 0
        else:
            prec_1 = prec_1.item()

        print('Best threshold for f1 score of target {}:'.format(j))
        print('Acc: {:.4f}, Recall_0: {:.4f}, Prec_0: {:.4f}, F1_0: {:.4f}, AUC: {:.4f}'.format(acc, recall_0, prec_0, f1_0, auc))
        print('             Recall_1: {:.4f}, Prec_1: {:.4f}, F1_1: {:.4f}\n'.format(recall_1, prec_1, f1_1))

        cm = confusion_matrixs[i]
        print('         |  Pred 0  |  Pred 1 ')
        print('------------------------------')
        print('Target 0 | {:8d} | {:8d}'.format(cm[0],cm[1]))
        print('------------------------------')
        print('Target 1 | {:8d} | {:8d}'.format(cm[2],cm[3])) 
        print('------------------------------\n\n')


for epoch in range(1, 15):

    loss = train(epoch, train_loader)
    print('\n\nEPOCH {}  TRAINING LOSS: {:.5f}\n'.format(epoch, loss))

    print('--- TESTING TRAIN DATA ---')
    test(train_loader)

    print('--- TESTING TEST DATA ---')
    test(test_loader)

    #torch.save(model.state_dict(), 'model_{:03d}.pt'.format(epoch))
