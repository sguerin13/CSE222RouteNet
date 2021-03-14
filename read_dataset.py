from torch.utils.data import Dataset, DataLoader
import numpy as np

from datanetAPI import DatanetAPI


class NetworkDataset(Dataset):
    def __init__(self, path, shuffle=False):
        super(NetworkDataset, self).__init__()

        reader = DatanetAPI(path, [], shuffle)
        it = iter(reader)
        self.data = list(it)
#         print(self.data)
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # reference: https://github.com/knowledgedefinednetworking/RouteNet-challenge/blob/master/code/read_dataset.py
        sample = self.data[index]

        ###################
        #  EXTRACT PATHS  #
        ###################
        routing = sample.get_routing_matrix()

        nodes = len(routing)
        # Remove diagonal from matrix
        paths = routing[~np.eye(routing.shape[0], dtype=bool)].reshape(
            routing.shape[0], -1)
        paths = paths.flatten()

        ###################
        #  EXTRACT LINKS  #
        ###################
        g = sample.get_topology_object()

        # Initialize with shape and value None
        cap_mat = np.full(
            (g.number_of_nodes(), g.number_of_nodes()), fill_value=None)

        for node in range(g.number_of_nodes()):
            for adj in g[node]:
                cap_mat[node, adj] = g[node][adj][0]['bandwidth']

#         print(cap_mat)

        links = np.where(np.ravel(cap_mat) != None)[0].tolist()

        link_capacities = (np.ravel(cap_mat)[links]).tolist()

        ids = list(range(len(links)))
        links_id = dict(zip(links, ids))

        path_ids = []
        for path in paths:
            new_path = []
            for i in range(0, len(path) - 1):
                src = path[i]
                dst = path[i + 1]
                new_path.append(links_id[src * nodes + dst])
            path_ids.append(new_path)

        ###################
        #   MAKE INDICES  #
        ###################
        link_indices = []
        path_indices = []
        sequ_indices = []
        segment = 0
        for p in path_ids:
            link_indices += p
            path_indices += len(p) * [segment]
            sequ_indices += list(range(len(p)))
            segment += 1

        traffic = sample.get_traffic_matrix()
        # Remove diagonal from matrix
        traffic = traffic[~np.eye(traffic.shape[0], dtype=bool)].reshape(
            traffic.shape[0], -1)

        result = sample.get_performance_matrix()
        # Remove diagonal from matrix
        result = result[~np.eye(result.shape[0], dtype=bool)].reshape(
            result.shape[0], -1)

        avg_bw = []
        pkts_gen = []
        delay = []
        jitter = []
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                flow = traffic[i, j]['Flows'][0]
                avg_bw.append(flow['AvgBw'])
                pkts_gen.append(flow['PktsGen'])
                d = result[i, j]['AggInfo']['AvgDelay']
                j = result[i,j]['AggInfo']['Jitter']
                delay.append(d)
                jitter.append(j)

        n_paths = len(path_ids)
        n_links = max(max(path_ids)) + 1

        return {
            "bandwith": avg_bw,
            "packets": pkts_gen,
            "link_capacity": link_capacities,
            "links": link_indices,
            "paths": path_indices,
            "sequences": sequ_indices,
            "n_links": n_links,
            "n_paths": n_paths
        }, (delay,jitter)


def get_dataloader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size, shuffle)


if __name__ == "__main__":
    path = '~/Documents/UCSD/222A/project/RouteNet-challenge/data/sample_data/test'
    # path = '~/222A/RouteNet-challenge/data/sample_data/test'
    dataset = NetworkDataset(path)

    dataloader = get_dataloader(dataset, 8)

    for batch_index, batch in enumerate(dataloader):
        x, delay = batch
