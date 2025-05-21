'''
Created on Jul 25, 2022

Compute the loss function

@author: masoud
'''
import tensorflow as tf
from tensorflow.keras.losses import Loss

class Weighted_DCG_ListMLE_topK(Loss):
    def __init__(self):
        super(Weighted_DCG_ListMLE_topK, self).__init__()             
    def call (self,y_true,y_pred):
        '''
            @param y_true: indexes of Top-k nodes + dcg_vals
            @param y_pred: output of the DNN
        '''
        raw_max = tf.reduce_max(input_tensor=y_pred, axis=1, keepdims=True)
        y_pred = y_pred - raw_max   
        y_true, DCG = tf.split(y_true, [y_true.shape[1]-1,1], axis=1) ## Split the y_true to obtain the topK and dcg_vals 
        y_true = tf.cast(y_true, dtype='int32') # cast back the labels to int
        sum_all = tf.reduce_sum(input_tensor=tf.exp(y_pred), axis=1, keepdims=True) # summation of exp(x) for all values
        y_ture_scores = tf.gather(y_pred,y_true,axis=1,batch_dims=1) # Fetch the similarity scores of topK nodes
        cumsum_y_ture_scores = tf.cumsum(tf.exp(y_ture_scores), axis=1, reverse=False, exclusive=True) # cumulative sum for exp of y_ture_scores 
        final_sum = sum_all - cumsum_y_ture_scores        
        loss_values = DCG * (tf.math.log(tf.math.abs(final_sum) + tf.keras.backend.epsilon()) - y_ture_scores) # loss values per each topK node        
        negative_log_likelihood = tf.reduce_sum(input_tensor=loss_values, axis=1, keepdims=True)
        return negative_log_likelihood