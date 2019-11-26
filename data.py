import json
import os.path as osp

# remove later
import numpy

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
#    index = torch.tensor(indices, dtype=torch.long)
#    assoc = torch.full((index.max() + 1, ), -1, dtype=torch.long)
#    assoc[index] = torch.arange(index.size(0))
#    return assoc

    assoc = {} # catch KeyError
    count = 0
    for i in indices:
        assoc[i] = count
        count += 1
    return assoc


def from_assoc(assoc, idx):
    try:
        return assoc[idx]
    except KeyError:
        return -1


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
    print('Vertices', vertex.size(), len(vertex_assoc))

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
    print('Particles', particle.size(), len(particle_assoc))

    clusters = []
    cluster_indices = []
    for key, item in data['VPClusters'].items():
        #drin = 0
        #for p in item['MCPs']:
        #    particle_id = from_assoc(particle_assoc, int(p))
        #    drin = max(drin, eta_mask[particle_id].item())
        #if drin:
            clusters.append([item['x'], item['y'], item['z']])
            cluster_indices.append(int(key))
    clusters = torch.tensor(clusters, dtype=torch.float)
    cluster_assoc = to_assoc(cluster_indices)
    cluster_indices = torch.tensor(cluster_indices)

    print('Clusters', clusters.size(), len(cluster_assoc))


    tracks = []
    track_indices = []
    for key, item in data['VeloTracks'].items():
        tracks.append([from_assoc(cluster_assoc, int(c)) for c in item['LHCbIDs']])
        track_indices.append(int(key))
    track_assoc = to_assoc(track_indices)
    print('Tracks', len(tracks), len(track_assoc))


    edge_index_0 = []
    edge_index_1 = []
    for track in tracks:
        for i in range(len(track) - 1):
            edge_index_0.append(track[i])
            edge_index_1.append(track[i+1])
    edge_index_tracks = torch.tensor([edge_index_0, edge_index_1])
    print('Edge Index Tracks:', edge_index_tracks.size())


    z = 2
    edge_index_0 = []
    edge_index_1 = []
    zs = []
    for i in range(len(clusters)):
        z_ci = clusters[i][z]
        if z_ci in zs:
            continue
        else:
            zs.append(z_ci)
            clusters_z_ci = cluster_indices[clusters[:,z] == z_ci]

            for j in range(len(clusters_z_ci)):
                for k in range(len(clusters_z_ci)):
                    if j != k:
                        edge_index_0.append(from_assoc(cluster_assoc, cluster_indices[j]))
                        edge_index_1.append(from_assoc(cluster_assoc, cluster_indices[k]))
    edge_index_z = torch.tensor([edge_index_0, edge_index_1])
    print('Edge Index Z:', edge_index_z.size())


    # visualize(clusters, vertex, particle, tracks)


if __name__ == '__main__':
#    path = '~/Desktop/LHC-data/data_2/VPData_6718814_1301.json'
    path = '/data/lhcb/lhcb_data/data_2/VPData_6718814_1301.json'
    path = osp.expanduser(osp.normpath(path))
    read_data(path)
