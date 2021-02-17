import Runner
import torch
import torch.nn as nn


'''
It could
be a generic end-to-end traffic flow. For instance, an MPLS
tunnel. Based on this assumption, RouteNet is built upon the
following principles:
1) The state of a path depends on the state of all the links
that lie on the path.
2) The state of a link depends on the state of all the paths
that traverse the link.
In a more formal description, let the state of a link be denoted
by hli, which is an unknown hidden vector. Similarly, the
state of a path is defined by hpi
. These principles can bemathematically formulated with the following expressions:


'''

class RouteNet(nn.Module):

    def __init__(self):
        super(RouteNet,self)__init__()

        ### Architecture ###

    def forward():
        pass