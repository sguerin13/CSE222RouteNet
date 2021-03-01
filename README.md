References:

Key Repo:

https://github.com/knowledgedefinednetworking

https://github.com/filipkrasniqi/QoSML

https://github.com/ITU-AI-ML-in-5G-Challenge/PS-014.2-GNN-Challenge-Gradient-Ascent/tree/master/code


RouteNet Sample Data:

https://github.com/knowledgedefinednetworking/RouteNet-challenge/tree/master/data/sample_data

Routenet Dataset Loader:

Using datanet api: https://github.com/knowledgedefinednetworking/RouteNet-challenge/blob/master/code/read_dataset.py

https://github.com/filipkrasniqi/QoSML/blob/master/supervised-learning-qos-learning/libs/routenet_dataset.py

https://github.com/filipkrasniqi/QoSML/blob/7efa4f5cbefe9b96a0888f4485b835ba0e719375/supervised-learning-qos-learning/libs/dataset_container.py

https://github.com/krzysztofrusek/net2vec/blob/2bed27f7db3114c366535a1deb2c8ec0f1fe13ef/routenet/upcdataset.py

Competition Data Page:

https://challenge.bnn.upc.edu/dataset



Explanation of Data Pipeline:

Sample:
- Routing Matrix: n x n - number of nodes
  - Each entry is a list containing the routers visited on path from src to dest where the src is the row and dest is the column
  - the diagonal is removed and it is flattened to a single vector. In the first data sample, the network has 14 nodes, so the vector is (14*14 - 14 x 1) = (182,1)

- Capacity Matrix: n x n
  - Each entry is non-zero only if there is a physical link between nodes, if non-zero, has the links capacity (bps)
  - Nonzero entries removed and flattened into a vector of size (n_links x 1) - 42 links in the first sample
  - Also create a new vector indexing the links [0,1,2....., 41]

- Path_ids: n_paths x 1 = 182 x 1
  - Takes the Routing Matrix and transforms it from a list where each entry is a list of nodes, to a list when each entry is a list of links.
    - i.e. [[0,1,2],[1,2,3]] -> [[0,1],[1,2]] where link 0 is the link between nodes 0 and 1

- Link, Path, Seq indices
  - Link indices:
    - take path_ids and flatten it from a list with nested lists to a single vector [390 x 1] in the first sample - [0,1,1,2]
  - Path indices
    - States which path each link belongs to in the link indices list [390 x 1] -> [0,0,1,1....]
  - Seq indices
    - States the order that each link is encounter along the path [390 x 1] -> [0,1,0,1....]
      - For example, the second entry in link indices (1) is the second link encounter on path 0.

- Result Data:
  - Traffic Matrix [n x n]
  - Delay Matrix [n x n]
  - both have the diagonal removed and are flattened to size [n_paths x 1] -> 182 x 1

Model Inputs:
- Link,Path,Seq Indices, n_paths, n_links

Model:
- Link hidden state matrix: n_links x hidden_state_dim
- Path hidden state matrix: n_paths x hidden_state_dim

- Message Passing:
  - The double for loop is aggregated into a single matrix of size H_links_T [390 x 32] - this is for the hidden state of all links encountered on all paths
    - the 390 comes from the double sum -> sum_over_all_path(sum_of_links_on_path_P_i)
      - some links will be repeated since there are only 42 discrete links and 390 links traversed between all paths
  
  - Path_Update_RNN_Input:
    - size (n_paths,max_len_path,link_hidden_state)
      - map values from H_links_T to this RNN input using keys from the path and sequence indices
  
  - Link Update:
    - The intermediate hidden states from the Path RNN update are aggregated into a message for the link state update
    - the path update hidden states are size (182,max_len_path,link_hidden_state)
      - they are aggregated into a size (n_links, link_hidden_state) size matrix by using a gather function
    - After aggregation the link state is updated by passing it through an RNN


