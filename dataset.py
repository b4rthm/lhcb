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

        data_list_pos = []
        data_list_neg = []
        i = 0
        for path in sorted(paths):
            if i % 100 == 0:
                print(i, '/', len(paths))
            i += 1
            data = self.process_example(path)

            if data is None:
                continue
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            if data.y == 1:
                data_list_pos.append(data)
            else:
                data_list_neg.append(data)

        data_list_pos.extend(data_list_neg)
        torch.save(self.collate(data_list_pos), self.processed_paths[0])

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
#
#            for particle in item['MCPs']:
#                eta_value = eta[particle_assoc[int(particle)]].item()
#                if (eta_value >= 2) and (eta_value <= 5):
#                    is_valid = True
#
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
           tracks.append([cluster_assoc[int(c)].item() for c in item['LHCbIDs']])
           track_indices.append(int(key))
        track_assoc = to_assoc(track_indices)

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
                 # Selfloops?
                 if j != k:
                   edge_index_0.append(cluster_assoc[cluster_indices[j]])
                   edge_index_1.append(cluster_assoc[cluster_indices[k]])
        edge_index_z = torch.tensor([edge_index_0, edge_index_1])

        return Data(pos=clusters, y=y, edge_index_tracks=edge_index_tracks, edge_index_z=edge_index_z)

if __name__ == '__main__':
    dataset = LHCbDataset(root='LHC-data')
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=6)
    print(dataset)
    print(dataset.data.y.sum().item() / len(dataset))


    for data in loader:
        print(data)
        break
