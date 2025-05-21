#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

#from absl import app
#from absl import flags
import sys
import os
from data_loader import dataloader
from split import link_split
import odin.data_util as data_util
from odin.trainer import Trainer
from utils import init_seed
import argparse


parser = argparse.ArgumentParser(description="parameters")
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
parser.add_argument('--pool', type=int, default=10, help='Pool for negative sampling.')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID.')
parser.add_argument('--embedding_size', type=int, default=20, help='Embedding size for embedding based models.')
parser.add_argument('--epochs', type=int, default=200, help='Max epochs for training.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size.')
parser.add_argument('--neg_sample_rate', type=int, default=4, help='Negative Sampling Ratio.')
parser.add_argument('--num_workers', type=int, default=20, help='Number of processes for training and testing.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-8, help='Weight decay.')
parser.add_argument('--disen_weight', type=float, default=0.5, help='Weight for the disentanglement loss.')
parser.add_argument('--edge_weight', type=float, default=1.0, help='Weight for edge score.')
parser.add_argument('--use_gpu', type=bool, default=False, help='Use GPU or not.')
parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle the training set')
parser.add_argument('--option', type=str, default='odin', help='Training options.')

# Dataset
parser.add_argument('--dataset', type=str, default='cora_ml', help='Dataset name.')
parser.add_argument('--prob_val', type=float, default=0.05, help='Validation set proportion.')
parser.add_argument('--prob_test', type=float, default=0.15, help='Test set proportion.')
parser.add_argument('--runs', type=int, default=10, help='Number of runs.')
parser.add_argument('--input_file', type=str, default='../data/', help='Input file path.')
parser.add_argument('--emb_file', type=str, default='./odin/embedding/', help='Embeddings output filename.')
args = parser.parse_args()
init_seed(args.seed)

def main(args):
    dataset = dataloader(args.dataset)
    data = dataset[0]

    #data splitting
    link_data = link_split(
        name=args.dataset, 
        data=data, 
        num_splits=args.runs, 
        prob_test=args.prob_test, 
        prob_val=args.prob_val,  
        maintain_connect=True
    )
    
    emb_file_0 = args.emb_file
    for split in range(args.runs):
        #The observed graph
        print("Generate Embedding", split+1)
        edge_index = link_data[split]['graph']
        emb_file = os.path.join(emb_file_0, args.dataset)
        emb_file = os.path.join(emb_file, str(split))
        args.emb_file = emb_file

        dm = data_util.DatasetManager(args)
        dm.get_dataset_info(edge_index)
    
        trainer = Trainer(args, dm)
        trainer.train()

if __name__ == "__main__":
    main(args)

