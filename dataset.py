import os
import glob
import json

import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader

from data import to_assoc


class LHCbDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(LHCbDataset, self).__init__(root, transform, pre_transform,
                                          pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data_{}'.format(i) for i in range(1, 11)]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        raise RuntimeError('Raw data not found')

    def process(self):
        print('process')
        paths = glob.glob('{}{}*{}*.json'.format(self.raw_dir, os.sep, os.sep))

        data_list = []
        i = 0
        for path in sorted(paths):
            print(i, '/', len(paths))
            i += 1
            data = self.process_example(path, 1)
            if data is not None:
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        i = 0
        for path in sorted(paths):
            print(i, '/', len(paths))
            i += 1
            data = self.process_example(path, 0)
            if data is not None:
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)


        torch.save(self.collate(data_list), self.processed_paths[0])

    # process example with desired label d_y
    def process_example(self, path, d_y):
        with open(path, 'r') as f:
            obj = json.load(f)

        y = 1 if obj['BCount'] > 0 else 0

        if not y == d_y:
            return None
        print(y)
        y = torch.tensor([y], dtype=torch.long)

        etas = []
        particle_indices = []
        for key, item in obj['MCParticles'].items():
            etas.append(item['eta'])
            particle_indices.append(int(key))
        particle_assoc = to_assoc(particle_indices)
        eta = torch.tensor(etas, dtype=torch.float)

        positions = []
        for key, item in obj['VPClusters'].items():
            is_valid = False
            for particle in item['MCPs']:
                eta_value = eta[particle_assoc[int(particle)]].item()
                if (eta_value >= 2) and (eta_value <= 5):
                    is_valid = True
            if is_valid:
                positions.append([item['z'], item['y'], item['x']])
        pos = torch.tensor(positions, dtype=torch.float)

        return Data(pos=pos, y=y)

if __name__ == '__main__':
    dataset = LHCbDataset(root='LHC-data')
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=6)
    print(dataset)
    print(dataset.data.y.sum().item() / len(dataset))


    for data in loader:
        print(data)
        break
