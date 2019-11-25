import time
import numpy

import os
import os.path as osp
import glob
import json

import torch
from torch_geometric.data import Dataset, Data, DataLoader

from data import to_assoc, from_assoc

start_time = time.time()

class LHCbDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(LHCbDataset, self).__init__(root, transform, pre_transform,
                                          pre_filter)

    @property
    def raw_file_names(self):
        return ['data_{}'.format(i) for i in range(1, 11)]

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(i) for i in range(0, 25)]

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        raise RuntimeError('Raw data not found')

    def process(self):
        print('process')
        paths = glob.glob('{}{}*{}*.json'.format(self.raw_dir, os.sep, os.sep))

        i = 0
        data_pos = []
        for path in sorted(paths):
            print(i, '/', len(paths),'Time elapsed: {:.1f}h'.format((time.time() - start_time)/3600))

            data = self.process_example(path)
            if data is None:
                continue
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if data.y == 0:
                torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
                i += 1
            else:
                data_pos.append(data)

        print('Negative Examples', i)
        print('Positive Examples', len(data_pos))

        for data in data_pos:
            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def process_example(self, path):
        with open(path, 'r') as f:
            obj = json.load(f)

        y = 1 if obj['BCount'] > 0 else 0
        y = torch.tensor([y], dtype=torch.long)

        etas = []
        particle_indices = []
        for key, item in obj['MCParticles'].items():
            etas.append(item['eta'])
            particle_indices.append(int(key))
        particle_assoc = to_assoc(particle_indices)
        eta = torch.tensor(etas, dtype=torch.float)

        clusters = []
        cluster_indices = []
        for key, item in obj['VPClusters'].items():
#            is_valid = False

#            for particle in item['MCPs']:
#                eta_value = eta[from_assoc(particle_assoc, int(particle))].item()
#                if (eta_value >= 2) and (eta_value <= 5):
#                    is_valid = True

#            if is_valid:
                clusters.append([item['z'], item['y'], item['x']])
                cluster_indices.append(int(key))
        cluster_assoc = to_assoc(cluster_indices)
        cluster_indices = torch.tensor(cluster_indices)

        clusters = torch.tensor(clusters, dtype=torch.float)
        if clusters.size(0) == 0:
           return None

        tracks = []
        track_indices = []
        for key, item in obj['VeloTracks'].items():
           tracks.append([from_assoc(cluster_assoc, int(c)) for c in item['LHCbIDs']])
           track_indices.append(int(key))

        edge_index_0 = []
        edge_index_1 = []
        for track in tracks:
            for i in range(len(track) - 1):
                edge_index_0.append(track[i])
                edge_index_1.append(track[i+1])
        edge_index_tracks = torch.tensor([edge_index_0, edge_index_1])

        edge_index_0 = []
        edge_index_1 = []
        zs = []
        for i in range(len(clusters)):
          z_ci = clusters[i][0]
          if z_ci in zs:
            continue
          else:
            zs.append(z_ci)
            clusters_z_ci = cluster_indices[clusters[:,0] == z_ci]

            for j in range(len(clusters_z_ci)):
              for k in range(len(clusters_z_ci)):
#                 if j != k:    # Selfloops?
                   edge_index_0.append(from_assoc(cluster_assoc, cluster_indices[j]))
                   edge_index_1.append(from_assoc(cluster_assoc, cluster_indices[k]))
        edge_index_z = torch.tensor([edge_index_0, edge_index_1])
        return Data(pos=clusters, y=y, edge_index_tracks=edge_index_tracks, edge_index_z=edge_index_z)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

if __name__ == '__main__':
    dataset = LHCbDataset(root='LHC-data')
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=6)
    print(dataset)

    for data in loader:
        print(data)
        break
