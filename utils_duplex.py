import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import os
import sys
import dgl
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import networkx as nx

def unique(x, dim=None):
    """
    Returns the unique elements of x along with the indices of those unique elements.

    Parameters:
    - x: Input tensor.
    - dim: Dimension along which to compute uniqueness. If None, the uniqueness is computed over the entire tensor.

    Returns:
    - unique: Tensor containing the unique elements of x.
    - inverse: Indices of the unique elements in the original tensor x.

    Reference:
    - https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810
    """
    
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)


def predictor(all_edges, am_outputs, ph_outputs, device):
    """
    Perform edge prediction based on input features and task type.

    Args:
    - all_edges (torch.Tensor): Tensor containing all edge pairs in the format (source, destination).
    - am_outputs (torch.Tensor): Tensor containing amplitude outputs.
    - ph_outputs (torch.Tensor): Tensor containing phase outputs.
    - task (int): Specifies the task to perform. 1 for existence prediction, 2 for direction prediction, 3 for three-type classification and 4 for four-type classification.
    - device (torch.device): Device to perform computations on.

    Returns:
    - neg_predicted (torch.Tensor): Negative of the predicted scores for each edge type.
    - labels (torch.Tensor): Predicted labels for the edge pairs.

    Note:
    The predicted labels are as follows:
    - 0: Reverse edges (rev)
    - 1: Positive edges (pos)
    - 2: Bidirectional edges (bi)
    - 3: Non-existent edges (non)
    """

    cos = torch.cos(ph_outputs[all_edges[:,0]] - ph_outputs[all_edges[:,1]]) # N*d
    sin = torch.sin(ph_outputs[all_edges[:,0]] - ph_outputs[all_edges[:,1]])
    mul = am_outputs[all_edges[:,0]].mul(am_outputs[all_edges[:,1]])
    real_part = mul.mul(cos).sum(dim=1)
    img_part = mul.mul(sin).sum(dim=1)

    predicted = torch.zeros(len(all_edges),4).to(device)
    predicted[:,0] = torch.abs(real_part)+torch.abs(img_part+1) # rev
    predicted[:,1] = torch.abs(real_part)+torch.abs(img_part-1) # pos
    predicted[:,2] = torch.abs(real_part-1)+torch.abs(img_part) # bi
    predicted[:,3] = torch.abs(real_part)+torch.abs(img_part) # non
    
    labels = torch.argmin(predicted,dim=1)

    #import ipdb; ipdb.set_trace()
    #the probabilities
    probabilities = F.softmax(-predicted, dim=1) 
    #the probability: u->v exitences an edge
    existence_probability = probabilities[:, 1] + probabilities[:, 2]


    
    return labels, existence_probability

class fourClassLoss(nn.Module):
    """
    Custom loss function for four-class classification tasks.

    Args:
    - args (Namespace): Arguments containing necessary parameters.
    
    Attributes:
    - softmax (nn.Softmax): Softmax layer for probability calculation.
    - celoss (nn.CrossEntropyLoss): Cross entropy loss with class weights.
    - bceloss (nn.BCEWithLogitsLoss): Binary cross entropy loss with logits.

    Methods:
    - forward(all_edges, am_outputs, ph_outputs, loss_weight): Computes the loss based on input data and loss weight.

    Note:
    - idx_1: Reverse edges
    - idx_2: Positive edges
    - idx_3: Bidirectional edges
    - idx_4: Non-existent edges
    """

    def __init__(self, args):
        super(fourClassLoss, self).__init__()
        self.args = args
        self.softmax = nn.Softmax(dim=1)
        self.celoss = nn.CrossEntropyLoss(weight = torch.tensor([1.0,1.0,1.0,1.0]))
        self.bceloss = nn.BCEWithLogitsLoss()


    def forward(self, all_edges, am_outputs, ph_outputs, loss_weight, origin_feat):
        """
        Compute the loss based on input data and loss weight.

        Args:
        - all_edges (torch.Tensor): Tensor containing all edge pairs in the format (source, destination, label).
        - am_outputs (torch.Tensor): Tensor containing amplitude outputs.
        - ph_outputs (torch.Tensor): Tensor containing phase outputs.
        - loss_weight (float): Weight for the connection-aware loss.

        Returns:
        - loss (torch.Tensor): Total loss computed based on input data and loss weight.
        """
        
        """
        Shapes:
        - all_edges: bz x 3
        """
        
        idx_1 = (all_edges[:,2] == 0) #rev
        idx_2 = (all_edges[:,2] == 1) #pos
        idx_3 = (all_edges[:,2] == 2) #bi
        idx_4 = (all_edges[:,2] == 3) #non

        cos = torch.cos(ph_outputs[all_edges[:,0]] - ph_outputs[all_edges[:,1]]) # N*d
        sin = torch.sin(ph_outputs[all_edges[:,0]] - ph_outputs[all_edges[:,1]])

        mul = am_outputs[all_edges[:,0]].mul(am_outputs[all_edges[:,1]])
        real_part = mul.mul(cos).sum(dim=1)
        img_part = mul.mul(sin).sum(dim=1)

        
        #for e in range(len(all_edges)):
        #    if ((real_part[e])**2+(img_part[e])**2)<1e-20:
        #        print(all_edges[e])
        #Slashdot Epinions original feat BLOOM

        # --------- connection aware loss ------------
        bi_predict = mul.sum(dim=1)
        ex_target = torch.ones(len(all_edges)).to(self.args.device)
        ex_target[idx_4] = 0.0
        exist_loss = self.bceloss(bi_predict, ex_target)
        ## ------------ END --------------
     
        predicted_2 = torch.zeros(len(all_edges),4).to(self.args.device)
        predicted_2[:,0] = - torch.sqrt((real_part)**2+(img_part+1)**2) # rev
        predicted_2[:,1] = - torch.sqrt((real_part)**2+(img_part-1)**2) # pos
        predicted_2[:,2] = - torch.sqrt((real_part-1)**2+(img_part)**2) # bi  
        #predicted_2[:,3] = - torch.sqrt((real_part)**2+(img_part)**2)  #non
        predicted_2[:, 3] = -torch.sqrt(torch.clamp(real_part**2 + img_part**2 + 1e-8, min=0.0))
        '''if origin_feat:
            #Avoiding NAN
            epsilon = 1e-8
            predicted_2[:, 3] = -torch.sqrt(torch.clamp(real_part**2 + img_part**2 + epsilon, min=0.0))
            #predicted_2[:,3] = - torch.sqrt((real_part)**2+(img_part)**2+ epsilon) # non
        else:
            predicted_2[:,3] = - torch.sqrt((real_part)**2+(img_part)**2)'''
        
        
        di_target = torch.zeros((len(all_edges), 4)).to(self.args.device)
        di_target[:,0] = idx_1.to(float) 
        di_target[:,1] = idx_2.to(float) 
        di_target[:,2] = idx_3.to(float) 
        di_target[:,3] = idx_4.to(float) 

        loss = self.celoss(predicted_2, di_target) + loss_weight*exist_loss
        

        #import ipdb; ipdb.set_trace()
        return loss


def load_training_data(args, data, feat, num_nodes, split):
    #num_nodes = feat.shape[0]
    edge_index = data['graph'] #The observed graph
    
    # DGL train graph
    train_graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    if feat is not None:
        train_graph.ndata['h'] = feat
    else:
        feat_file = f"./data/duplex/{args.dataset}/feat.txt"
        if os.path.exists(feat_file):
            feat = np.loadtxt(feat_file, delimiter=',', dtype=np.float32)
        else:
            feat = np.random.normal(0,1,(num_nodes, args.input_dim*2)).astype(np.float32)
            os.makedirs(os.path.dirname(feat_file), exist_ok=True)
            np.savetxt(feat_file, feat, fmt='%f', delimiter=',')
        train_graph.ndata['h'] = torch.tensor(feat)

    # Training edges
    training_file = f"./data/duplex/{args.dataset}/training_{split}.pt"
    if os.path.exists(training_file):
        print("load training edges")
        train_edges = torch.load(training_file)
    else:
        train_pos = data['train']['pos']
        train_neg = data['train']['neg']
        pos_edges = set(tuple(edge.tolist()) for edge in train_pos)
        undirected = []
        directed = set()
        for edge in pos_edges:
            a, b = edge
            if (b, a) in pos_edges:
                undirected.append([a, b, 2])
            else:
                directed.add(edge)
        undirected_tensor = torch.tensor(undirected, dtype=torch.long)
        directed = torch.tensor(list(directed), dtype=torch.long)
        directed = directed [torch.randperm(directed.size(0))]
    
        half = directed.size(0) // 2
        first_half_labeled = torch.cat([directed[:half], torch.ones((directed[:half].size(0), 1), dtype=torch.long)], dim=1)
        second_half_reversed = directed[half:][:, [1, 0]]
        second_half_labeled = torch.cat([second_half_reversed, torch.zeros((second_half_reversed.size(0), 1), dtype=torch.long)], dim=1)
        processed_directed = torch.cat([first_half_labeled, second_half_labeled], dim=0)
    
        train_neg_clean = []
        for edge in train_neg:
            a, b = edge.tolist()
            if (b, a) not in pos_edges:
                train_neg_clean.append([a, b])

        train_neg_clean = torch.tensor(train_neg_clean, dtype=torch.long)
        train_neg_labeled = torch.cat([train_neg_clean, torch.full((train_neg_clean.size(0), 1), 3, dtype=torch.long)], dim=1)  
        
        train_edges = torch.cat([undirected_tensor, processed_directed, train_neg_labeled], dim=0)
        os.makedirs(os.path.dirname(training_file), exist_ok=True)
        torch.save(train_edges, training_file)

    #import ipdb; ipdb.set_trace()

    # 0,1,2,3: rev, pos, bi, non
    rev_train_edges = train_edges[(train_edges[:,2]==0)]  # rev
    true_train_edges = train_edges[(train_edges[:,2]==1)|(train_edges[:,2]==2)]   # pos or bi
    none_edges_train = train_edges[(train_edges[:,2]==3)]   # none-edge

    rev_eid_train = train_graph.edge_ids(rev_train_edges[:,1], rev_train_edges[:,0])
    pos_eid_train = train_graph.edge_ids(true_train_edges[:,0], true_train_edges[:,1])

    # Set edge labels (1,2) and existence indicators for training edges
    train_graph.edata['label'] = torch.tensor([-1]*train_graph.num_edges()) # -1
    train_graph.edata['label'][rev_eid_train] = rev_train_edges[:,2]  # 0 rev
    train_graph.edata['label'][pos_eid_train] = true_train_edges[:,2]   # 1/2 pos/bi

    train_graph.edata['exist'] = torch.tensor([1.]*train_graph.num_edges())
    _a = train_graph.reverse().edges()[0]
    _b = train_graph.reverse().edges()[1]
    train_graph = dgl.add_edges(train_graph, _a, _b) 
    train_graph.edata['exist'][train_graph.edata['exist'] == 0] = -1.

    train_graph = dgl.add_edges(train_graph, none_edges_train[:,0], none_edges_train[:,1])
    none_eid_train = train_graph.edge_ids(none_edges_train[:,0], none_edges_train[:,1])
    train_graph.edata['exist'][none_eid_train] = 0.

    train_graph.edata['am_exist']= torch.tensor([0.]*train_graph.num_edges())
    train_graph.edata['am_exist'][train_graph.edata['exist'] != 0] = 1.
    
    w_edges = torch.cat((train_graph.edges()[0].unsqueeze(1), 
                            train_graph.edges()[1].unsqueeze(1)
                        ), dim=1)

    _, idx = unique(w_edges,dim=0)
    mask = torch.ones(len(w_edges), dtype=torch.bool)
    mask[idx] = False
    dup = mask.nonzero(as_tuple=False).squeeze()
    train_graph.edata['am_exist'][dup] = 0.
    train_graph.edata['label'][none_eid_train] = none_edges_train[:,2]
    
    # Create DataLoader for edge prediction training   
    train_eid = torch.cat((pos_eid_train, rev_eid_train, none_eid_train))
    
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        dgl.dataloading.MultiLayerFullNeighborSampler(args.nlayers)
        )

    #add self_loop
    train_graph = dgl.add_self_loop(train_graph) 

    if args.device.type == 'cpu':
        num_workers = 0
        use_uva = False
    else:
        num_workers = 0
        use_uva = True
    
    dataloader = dgl.dataloading.DataLoader(
        train_graph, train_eid, sampler,
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=False, 
        num_workers=num_workers,
        device=args.device,
        use_uva =use_uva)

    return dataloader, train_graph

def load_testing_data(args, data, train_graph):
    
    val_pos = data['val']['pos']
    val_neg = data['val']['neg']

    test_pos = data['test']['pos']
    test_neg = data['test']['neg']

    #val_pos_nodes = val_pos.unique()
    #val_neg_nodes = val_neg.unique()
    #test_pos_nodes = test_pos.unique()
    #test_neg_nodes = test_neg.unique()

    # Sample blocks for testing nodes
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.nlayers)
    test_pos_nodes = test_pos.unique()
    
    _, _, val_pos_blocks = sampler.sample_blocks(train_graph, val_pos.unique().to(args.device))
    _, _, val_neg_blocks = sampler.sample_blocks(train_graph, val_neg.unique().to(args.device))
    _, _, test_pos_blocks = sampler.sample_blocks(train_graph, test_pos.unique().to(args.device))
    _, _, test_neg_blocks = sampler.sample_blocks(train_graph, test_neg.unique().to(args.device))

    return val_pos_blocks, val_neg_blocks, test_pos_blocks, test_neg_blocks



