import torch
import torch.nn as nn
import torch.nn.functional as F

'''using the implementation from RouteNet'''
class RouteNet(nn.Module):

    def __init__(self):
        super(RouteNet,self).__init__()

        ### Architecture ###
        # for gru need to pay attention to if input is of size:
        # (batch, seq_len, feature size) or seq_len, batch, feature size
        # if sequence length is variable
        # may need to pad the sequence
        self.link_state_dim = 512
        self.path_state_dim = 512
        self.readout_dim = 256
        self.output_units = 1
        self.T = 8
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        inSize = 10 # place holder
        hSize  = 512
        readSize = 10
        nLayer = 1

        self.l_U = nn.GRU(input_size = self.link_state_dim,
                          hidden_size = hSize,
                          num_layers = nLayer,
                          batch_first=True)
        
        self.p_U = nn.GRU(input_size = self.path_state_dim,
                          hidden_size = hSize,
                          num_layers = nLayer,
                          batch_first=True)
        
        self.readOut_Delay = nn.ModuleDict({'r1': nn.Linear(hSize,self.readout_dim),
                                            'r2': nn.Linear(self.readout_dim,int(self.readout_dim/2)),
                                            'r3': nn.Linear(int(self.readout_dim/2),self.output_units)
                                            })
        
        self.readOut_Jitter = nn.ModuleDict({'r1': nn.Linear(hSize,self.readout_dim),
                                             'r2': nn.Linear(self.readout_dim,int(self.readout_dim/2)),
                                             'r3': nn.Linear(int(self.readout_dim/2),self.output_units)
                                              })


    def forward(self,x):
        
        links = x['links']                      
        paths = x['paths']                
        seqs = x['sequences']
        link_cap = x['link_capacity']
        # link_cap = link_cap/torch.max(link_cap)
        bandwidth = x['bandwith']
        # bandwidth = bandwidth/torch.max(bandwidth)

        # state matrix shape for the link
        link_h_state_shape = (x['n_links'][0], self.link_state_dim-1)

        # create hidden state matrix shape for the path  
        path_h_state_shape = (x['n_paths'],self.path_state_dim-1)
        path_h_state = torch.cat((bandwidth,torch.zeros(path_h_state_shape).to(self.device)), axis=1)
        
        # prepare input for path update RNN
        max_seq_len = torch.max(seqs)
        path_rnn_input_shape = (x['n_paths'],max_seq_len+1,self.link_state_dim)
        
        #stack the paths and sequences
        ids = torch.stack((paths,seqs),axis=1)
        ids = torch.squeeze(ids,2)           
        p_ind = ids[:,0]
        s_ind = ids[:,1]

        # flatten the double loop into a bulk matrix using the gather functionality
        # this is an aggregation of the state vector of each link on each path flattened into a (sum_paths(sum_links_on_paths) x h_state) size matrix 
        # using torch.gather
        indices = torch.zeros(len(links),self.link_state_dim).to(self.device)
        for i in range(len(links)):
            link_id = links[i]
            indices[i,:] = link_id
            
        # variable dictionary for forward pass
        vd = {}
        
        for t in range(self.T):
            
            
            ############# set up the matrices and variables for each pass through #################
            
            ########## PATH VARIABLES ###########
            
            # input to the path rnn layer P_u
            path_rnn_input_key = 'path_rnn_input_' + str(t)
            vd[path_rnn_input_key] = torch.zeros(path_rnn_input_shape).to(self.device)
            
            if (t > 0):  # for non leaf variables, we need to propagate the gradient back
                vd[path_rnn_input_key].requires_grad = True
            
            
            # path hidden state output from P_U, initialized with just bandwidth at T_0, else copy
            path_h_state_key = 'path_h_state_' + str(t)
            if (t==0):
                vd[path_h_state_key] = torch.cat((bandwidth,torch.zeros(path_h_state_shape).to(self.device)), axis=1)
            else:
                path_h_state_key = 'path_h_state_' + str(t)
                path_h_prev = 'path_h_state_' + str(t-1)
                vd[path_h_state_key] = vd[path_h_prev]
            
            
            # path_hidden state sequence from P_U, used to update links
            path_h_state_seq_key = 'path_h_states_seq_' + str(t)
            
            
            ########## LINK VARIABLES ###########
            
            # vector to store the link_hidden states
            if (t == 0):
                # create hidden state matrix for links and initialize with first column as link capacity
                link_h_state_key = 'link_h_state_' + str(t)
                vd[link_h_state_key] = torch.cat((link_cap,torch.zeros(link_h_state_shape).to(self.device)),1)
            else:
                # copy hidden state value for next pass through
                link_h_state_key = 'link_h_state_' + str(t)
                linK_h_prev = 'link_h_state_' + str(t-1)
                vd[link_h_state_key] = vd[linK_h_prev]
                
                
            # matrix storing the hidden states of links on paths
            # i.e. the hidden state of all links in the x['links'] list
            h_link_path_key = 'h_links_on_paths_' + str(t)
            vd[h_link_path_key] = torch.gather(vd[link_h_state_key],0,indices.long())
            
            #link messages extracted from the path hidden state sequence output from P_U
            link_message_key = 'link_messages_' + str(t)

            
             # container for the link messages that are extracted from path rnn hidden states
            agg_link_message_key = 'aggregated_link_message_' + str(t)
            vd[agg_link_message_key] = torch.zeros((x['n_links'],self.link_state_dim),requires_grad=True).to(self.device)
            
            ########################################################################################
            
            
            ################################## DO THE MESSAGE PASSING ##############################
            
            # prepare input for path RNN
            vd[path_rnn_input_key] = vd[path_rnn_input_key].index_put(indices = [p_ind,s_ind],
                                                                      values = vd[h_link_path_key])
            # pass through the path RNN
            vd[path_h_state_seq_key], vd[path_h_state_key] = self.p_U(vd[path_rnn_input_key],
                                                                      torch.unsqueeze(vd[path_h_state_key],
                                                                      0))
            # reformat
            vd[path_h_state_key] = vd[path_h_state_key].squeeze(0)
            
            # extract link messages from the path RNN sequence output
            # equivalent to tf.gather_nd
            vd[link_message_key] = vd[path_h_state_seq_key][p_ind,s_ind,:]
           
            # aggregate the link messages
            vd[agg_link_message_key] = vd[agg_link_message_key].index_put([links.squeeze(1)],
                                                                           vd[link_message_key],
                                                                           accumulate=True)
            # update the state of the links by passing through link 
            _, vd[link_h_state_key] = self.l_U(torch.unsqueeze(vd[agg_link_message_key],1),
                                               torch.unsqueeze(vd[link_h_state_key].squeeze(0),0))
            # reformat
            vd[link_h_state_key] = vd[link_h_state_key].squeeze(0)
            
            ##########################################################################################

        # readout from the paths
        d,j = self.readout(vd[path_h_state_key])
        return d,j


    def readout(self,path_state):
        d = F.relu(self.readOut_Delay['r1'](path_state))
        d = F.relu(self.readOut_Delay['r2'](d))
        d = self.readOut_Delay['r3'](d)

        j = F.relu(self.readOut_Jitter['r1'](path_state))
        j = F.relu(self.readOut_Jitter['r2'](j))
        j = self.readOut_Jitter['r3'](j)
        
        return (d,j)