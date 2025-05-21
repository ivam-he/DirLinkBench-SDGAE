'''
Created on March 21, 2023
Implements Matrix form of AP-scores 
@author: masoud
'''
import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
import eltra.fetch_topK_DENSE as prData
from scipy.sparse import identity 

def AP_Scores_ (edge_index, iterations=0, damping_factor=0.8,  impr_fact = 0.80, topK=0, directed_graph=True, loss=''):
    '''
        Computes AP-scores:
            1- An importance factor is assigned to downgrade in-links and upgrade out-links contribution in similarity computation     
            2- prx(a,b) = 0 if b \in I_a; otherwise prx(a,b)= CRW(a,b)    
            3- In the case of undirected graphs, Symmetric AP-socres are computed        
    '''
    if topK != -1:
        print("Computing AP-scores on '{}' itrations, top '{}', and C '{}'...".format(iterations,topK,damping_factor)+'\n')  
    else:
        print("Computing AP-scores on '{}' itrations, and C '{}'...".format(iterations,damping_factor)+'\n')
                       
    #============================================================================================
        # reading graph with networkX
    #============================================================================================    

    #from edge_index to netwoekX
    G = nx.DiGraph()
    num_nodes = edge_index.max().item() + 1
    G.add_nodes_from(range(num_nodes))
    for src, dst in zip(edge_index[0], edge_index[1]):
        G.add_edge(src.item(), dst.item())

    #G = nx.read_edgelist(graph, create_using=nx.DiGraph(), nodetype = int)
    
    nodes = sorted(G.nodes())       # sorted list of all nodes        
    adj = nx.adjacency_matrix(G,nodelist=nodes, weight=None)      # V*V adjacency matrix 
    print("Number of nodes in graph: ",len(nodes))
    if topK == -1: ## calculating topK for the input graph
        topK = round(G.number_of_edges()/G.number_of_nodes())*20    
        print("TopK is calculated and set as '{}' ...".format(topK)+'\n')
        if G.number_of_edges()<=topK: ## if graph has few number of nodes
            print('*** topK {} is larger than the number of nodes in the graph! set the topK value manually ***'.format(topK))
            print('')
            return 
    topK = topK + 1  ## a node itself is also considered in the topK list as the most similar one to itself               
    adj_normalized_inlink = normalize(adj, norm='l1', axis=0).T # column normalized adj    
    adj_normalized_outlink = normalize(adj, norm='l1', axis=1) # row normalized adj  
    result_matrix = identity(len(nodes),dtype=float)
    result_matrix = result_matrix.todense()
    for itr in range (1, iterations+1):           
        print("Iteration "+str(itr)+' ....')
        result_matrix =  damping_factor/2.0 * ( (1-impr_fact) * (adj_normalized_inlink * result_matrix) + ( impr_fact * adj_normalized_outlink * result_matrix) ) 
        np.fill_diagonal(result_matrix,1) ## S(a,a)=1 is the stop case for the recursive computaiton
        if not directed_graph: ## In the case of UNDIRECTED graphs; symmetric scores are considered.
            result_matrix = 0.5 * (result_matrix + result_matrix.T) 
            print('AP-scores are calculated as symmetric ....\n')  
            
    if loss=='weighted_listMLE_topK': 
        if directed_graph: ## if the graph is directed
            prox = np.multiply((~adj.toarray() + 2).T, result_matrix) ## computing PROX scores
            np.fill_diagonal(prox,1)
            top_indices,dcg_vals = prData.get_weighted_listMLE_topK(prox, topK)
            return top_indices,dcg_vals
        else: ## if the graph is undirected, top nodes are selected directly
            top_indices,dcg_vals = prData.get_weighted_listMLE_topK(result_matrix, topK)         
            return top_indices,dcg_vals


'''
AP_Scores_(graph="/home/masoud/backup_1/workspace/iterative-adamic/src/graphs/AdaSimStar_undirected.txt", 
           iterations=3, 
           topK=5, 
           damping_factor =0.6,
           directed_graph=False,
           loss='weighted_listMLE_topK'
           ) 


'''           

