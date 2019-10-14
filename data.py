import json
import os.path as osp

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def visualize(cluster, vertex, particle, tracks):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for track in tracks:
        for i in range(len(track) - 1):
            xs = [cluster[track[i], 2].item(), cluster[track[i + 1], 2].item()]
            ys = [cluster[track[i], 1].item(), cluster[track[i + 1], 1].item()]
            zs = [cluster[track[i], 0].item(), cluster[track[i + 1], 0].item()]
            ax.plot(xs, ys, zs, c='black')

    ax.scatter(cluster[:, 2], cluster[:, 1], cluster[:, 0], s=4, c='blue')
    ax.scatter(vertex[:, 2], vertex[:, 1], vertex[:, 0], s=4, c='red')
    ax.scatter(particle[:, 2], particle[:, 1], particle[:, 0], s=4, c='green')

    ax.set_xlabel('z')
    ax.set_ylabel('y')
    ax.set_zlabel('x')

    plt.show()


def to_assoc(indices):
    index = torch.tensor(indices, dtype=torch.long)
    assoc = torch.full((index.max() + 1, ), -1, dtype=torch.long)
    assoc[index] = torch.arange(index.size(0))
    return assoc


def read_data(path):
    with open(path, 'r') as f:
        data = json.load(f)

    vertices = []
    vertex_indices = []
    for key, item in data['MCVertices'].items():
        vertices.append(item['Pos'])
        vertex_indices.append(int(key))
    vertex = torch.tensor(vertices, dtype=torch.float)
    vertex_assoc = to_assoc(vertex_indices)
    print('Vertices', vertex.size(), vertex_assoc.size())

    particles = []
    particle_indices = []
    etas = []
    for key, item in data['MCParticles'].items():
        particles.append(item['OVPos'])
        particle_indices.append(int(key))
        etas.append(item['eta'])
    particle = torch.tensor(particles, dtype=torch.float)
    particle_assoc = to_assoc(particle_indices)
    eta = torch.tensor(etas, dtype=torch.float)
    eta_mask = (eta >= 2) & (eta <= 5)
    print('Particles', particle.size(), particle_assoc.size())

    clusters = []
    cluster_indices = []
    for key, item in data['VPClusters'].items():
        drin = 0
        for p in item['MCPs']:
#            print(p)
            particle_id = particle_assoc[int(p)]
            drin = max(drin, eta_mask[particle_id].item())
        if drin:
            clusters.append([item['x'], item['y'], item['z']])
            cluster_indices.append(int(key))
    cluster = torch.tensor(clusters, dtype=torch.float)
    cluster_assoc = to_assoc(cluster_indices)
    print('Clusters', cluster.size(), cluster_assoc.size())

    tracks = []
    track_indices = []
    for key, item in data['VeloTracks'].items():
        tracks.append([cluster_assoc[int(c)].item() for c in item['LHCbIDs']])
        track_indices.append(int(key))
    track_assoc = to_assoc(track_indices)
    print('Clusters', len(tracks), track_assoc.size())

    visualize(cluster, vertex, particle, tracks)


if __name__ == '__main__':
#    path = '~/Desktop/LHC-data/data_2/VPData_6718814_1301.json'
    path = '/data/lhcb/lhcb_data/data_2/VPData_6718814_1301.json'
    path = osp.expanduser(osp.normpath(path))
    read_data(path)
