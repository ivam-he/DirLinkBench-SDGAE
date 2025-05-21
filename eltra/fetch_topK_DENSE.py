'''
Created on Jul 7, 2022
@note Fetches topK similar nodes to each node from a similarity matrix 
@author: masoud
'''
import numpy as np

def get_weighted_listMLE_topK(result_matrix, topK):    
    '''
        prepares the required information for 'Weighted_DCG_ListMLE_topK' loss function
        
        @return: top_indices: |V|*topK matrix contians indices of topK nodes to each node
        @return: dcg_vals: |V|*|1| matrix contains Discounted Cumulative Gain (DCG) of sorted similarity scores w.r.t each node                 
    '''   
    print ("Preparing top similar nodes and DCG values ...")
    top_indices = np.zeros((result_matrix.shape[0],topK),dtype='float32')
    dcg_vals = np.zeros((result_matrix.shape[0],1),dtype='float32')
    for target_node in range (0,result_matrix.shape[0]):
        target_node_res_sorted = np.argsort(result_matrix[target_node], axis=1)[0,::-1][0,:topK] #[0,::-1][0,:topK]  ## sorting the indices on descending order of similairity values and return the topk
        top_indices[target_node] = target_node_res_sorted.copy()
        
        """ In computing DCG, scores are normally in a small integer range; we transform the AP-socres in range [0,10] """
        dcg_denominator = np.log2(np.arange(2, result_matrix.shape[1]+2))
        result_matrix[target_node] = -np.sort(-np.floor(result_matrix[target_node]*10),axis=1)   
        dcg_vals[target_node] = np.sum(np.divide(result_matrix[target_node],dcg_denominator))
                
    print ("Top {} similar nodes are fetched ... ".format(topK-1))
    return top_indices,dcg_vals
