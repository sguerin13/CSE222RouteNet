import Runner
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv

'''using the implementation from RouteNet'''

class RouteNet(nn.Module):

    def __init__(self):
        super(RouteNet,self).__init__()

        ### Architecture ###
        # for gru need to pay attention to if input is of size:
        # (batch, seq_len, feature size) or seq_len, batch, feature size
        # if sequence length is variable
        # may need to pad the sequence
        self.link_state_dim = 32
        self.path_state_dim = 32
        self.readout_dim = 8
        self.output_units = 1
        self.T = 8

        inSize = 10 # place holder
        hSize  = 32
        readSize = 10
        nLayer = 1

        # need to dial this in a bit
        self.l_U = nn.GRU(self.link_state_dim,hSize,nLayer,batch_first=True)
        self.p_U = nn.GRU(self.path_state_dim,hSize,nLayer,batch_first=True)

        self.readOut1 = nn.Linear(hSize,self.readout_dim)
        self.readOut2 = nn.Linear(self.readout_dim,self.readout_dim)
        self.readOut3 = nn.Linear(self.readout_dim,self.output_units)


    def forward(self,x):
        links = x['links']
        paths = x['paths']
        seqs = x['sequences']

        # state matrix shape for the link
        link_h_state_shape = torch.stack([x['n_links'], self.link_state_dim-1], axis = 1)

        # create hidden state matrix for links and initialize with first column as link capacity
        link_state = torch.cat([torch.unsqueeze(x['link_capacity'],axis=1),
                                torch.zeros(link_h_state_shape)])

        # state matrix shape for the path  
        path_h_state_shape = torch.stack([ x['n_paths'],self.path_state_dim-1], axis=0)

        # create hidden state matrix for path and initialize with first column as path bandwith
        path_state = torch.cat([torch.unsqueeze(x['bandwith'], axis=1),
                                torch.zeros(path_h_state_shape)], axis=1)

        for t in range(self.T):

            # flatten the double loop into a bulk matrix using the gather functionality
            # this is an aggregation of the state vector of each link on each path flattened into a (sum_paths(sum_links_on_paths) x h_state) size matrix 
            # https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
            indices = torch.zeros(len(links),32)
            for i in range(len(links)):
                link_id = links[i]
                indices[i,:] = link_id

            h_t_links_on_paths = torch.gather(link_h_state,0,indices.long())

            #stack the paths and sequences
            paths = torch.tensor(paths)
            seqs = torch.tensor(seqs)
            ids = torch.stack((paths,seqs),axis=1)

            # prepare input for path update RNN
            max_seq_len = torch.max(seqs)
            path_rnn_input_shape = (x['n_paths'],max_seq_len,self.link_state_dim)

            # aggregate the sequence lengths
            scatter_src = torch.unsqueeze(torch.ones_like(paths),1)
            lens = torch.zeros(x['n_paths'],1).long()
            _ = lens.scatter_add_(dim = 0, src = scatter_src,
                           index=torch.unsqueeze(paths,1))

            # move the elements from h_t_links on paths to the rnn input
            path_rnn_input = torch.zeros(path_rnn_input_shape)
            for i in range(ids.shape[0]):
                p_ind = ids[i,0]
                s_ind = ids[i,1]
                path_rnn_input[p_ind,s_ind,:] = h_t_links_on_paths[i]

        
            hidden_states, path_state = self.P_U(path_rnn_input,initial_state)

            # For every link, gather the input from the hidden states of the paths
            # outputs is size (n_path, max_seq, link_state_dim)  182 x 3 x 32
            # ids are size (sum_path(sum_link_in_path) x 2)      390 x 2
            # pytorch doesn't have a gather_nd instead it will grab by output
            link_messages = torch.zeros((len(links),self.link_state_dim))
            for i in range(ids.shape[0]):
                p_ind = ids[i,0]
                s_ind = ids[i,1]
                link_messages[i] = hidden_states[p_ind,s_ind,:]

            # aggregate the messages
            # aggregated link message is size (n_links x link state dim)
            # aggregated along rows based on link index in the links vector
            aggregated_link_message = torch.zeros((x['n_links'],link_state_dim))
            _ = aggregated_link_message.scatter_add_(0,torch.tensor(links).unsqueeze(1),link_messages)

            # update the state of the links
            link_state, _ = self.l_U(aggregated_link_message, [link_state])


        # readout from the paths
        y = self.readout(path_state)
        return y

    def readout(self,path_state):
        x = nn.ReLU(self.readOut1(path_state))
        x = nn.ReLU(self.readOut2(x))
        x = self.readOut3(x)
        return x

