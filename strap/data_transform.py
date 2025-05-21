import sys
import os
from data_loader import dataloader
from split import link_split
import argparse
from utils import init_seed

parser = argparse.ArgumentParser(description="parameters")
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
# Dataset
parser.add_argument('--dataset', type=str, default='cora_ml', help='Dataset name.')
parser.add_argument('--prob_val', type=float, default=0.05, help='Validation set proportion.')
parser.add_argument('--prob_test', type=float, default=0.15, help='Test set proportion.')
parser.add_argument('--runs', type=int, default=10, help='Number of runs.')
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

    save_dir = './strap/data'
    save_dir = os.path.join(save_dir, args.dataset)
    num_nodes = data.x.shape[0]

    for split in range(args.runs):
        #The observed graph
        save_path = os.path.join(save_dir, args.dataset+str(split)+'.txt')
        os.makedirs(save_dir, exist_ok=True)
        edge_index = link_data[split]['graph']

        with open(save_path, "w") as f:
            f.write(str(num_nodes) + "\n")
            for i in range(edge_index.size(1)):
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                f.write(f"{src} {dst}\n")

    print(f"{args.dataset} saved successfully to {save_dir}!")
        

if __name__ == "__main__":
    main(args)

