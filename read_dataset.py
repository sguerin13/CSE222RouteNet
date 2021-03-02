from torch.utils.data import Dataset, DataLoader
import numpy as np

from datanetAPI import DatanetAPI


class NetworkDataset(Dataset):
    def __init__(self, path, shuffle=False):
        super(NetworkDataset, self).__init__()
        
        reader = DatanetAPI(path, [], shuffle)
        it = iter(reader)
        self.data = list(it)
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
        paths = routing[~np.eye(routing.shape[0], dtype=bool)].reshape(routing.shape[0], -1)
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

        print(cap_mat)

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
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                flow = traffic[i, j]['Flows'][0]
                avg_bw.append(flow['AvgBw'])
                pkts_gen.append(flow['PktsGen'])
                d = result[i, j]['AggInfo']['AvgDelay']
                delay.append(d)

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
        }, delay

def get_dataloader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size, shuffle)   


if __name__ == "__main__":
    path = '~/Documents/UCSD/222A/project/RouteNet-challenge/data/sample_data/test'
    # path = '~/222A/RouteNet-challenge/data/sample_data/test'
    dataset = NetworkDataset(path)
    
    dataloader = get_dataloader(dataset, 8)
    
    for batch_index, batch in enumerate(dataloader):
        x, delay = batch
        
    

'''
README:
    Model Inputs:
    - Link,Path,Seq Indices, n_paths, n_links
    
TF version:
    ds = tf.data.Dataset.from_generator(
        lambda: generator(data_dir=data_dir, shuffle=shuffle),
        
        # output type
        ({"bandwith": tf.float32, "packets": tf.float32,
            "link_capacity": tf.float32, "links": tf.int64,
            "paths": tf.int64, "sequences": tf.int64,
            "n_links": tf.int64, "n_paths": tf.int64},
        tf.float32),
        
        # output shape
        ({"bandwith": tf.TensorShape([None]), "packets": tf.TensorShape([None]),
            "link_capacity": tf.TensorShape([None]),
            "links": tf.TensorShape([None]),
            "paths": tf.TensorShape([None]),
            "sequences": tf.TensorShape([None]),
            "n_links": tf.TensorShape([]),
            "n_paths": tf.TensorShape([])},
            tf.TensorShape([None])))


Sample:
- Routing Matrix: n x n - number of nodes

- Capacity Matrix: n x n

- Path_ids: n_paths x 1 = 182 x 1


- **Link indices**
    - take path_ids and flatten it from a list with nested lists to a single vector [390 x 1] in the first sample - [0,1,1,2]
- **Path indices**
    - States which path each link belongs to in the link indices list [390 x 1] -> [0,0,1,1....]
- **Seq indices**
    - States the order that each link is encounter along the path [390 x 1] -> [0,1,0,1....]
    - For example, the second entry in link indices (1) is the second link encounter on path 0.

- Result Data:
  - Traffic Matrix [n x n]
  - Delay Matrix [n x n]
  - both have the diagonal removed and are flattened to size [n_paths x 1] -> 182 x 1

Model Inputs:
- Link,Path,Seq Indices, n_paths, n_links

model.py have
x['n_paths']
x['n_links']

x['links']
x['link_capacity']

x['paths']
x['packets']

x['sequences']
x['bandwith']
'''
