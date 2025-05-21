import argparse
from itertools import product
import subprocess

def build_cmd(base_args, additional_args):
    args = base_args.copy()
    args.update(additional_args)
    cmd_parts = ["python main.py", 
                 "--net GAT",
                 "--origin_feat True",
                 "--hidden 8",
                 "--embedding single",
                 "--save_res True",
                 f"--dataset {args.pop('dataset')}",
                 f"--device {args.pop('device')}"]
    
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                value = 'True' if value else 'False'
                cmd_parts.append(f"--{key} {value}")
            else:
                pass
        else:
            cmd_parts.append(f"--{key} {value}")
    
    return " ".join(cmd_parts)

def main():
    parser = argparse.ArgumentParser(description="search")
    parser.add_argument('--dataset', type=str, default='cora_ml', help='Dataset name.')
    parser.add_argument('--device', type=int, default=0, help='Device ID.')
    args = parser.parse_args()

    dataset = args.dataset
    device = args.device

    undirecteds = [True, False]
    predictors = ["MLPScore", "InnerProduct"]
    operators = ["cat", "hadamard"]
    loss_functions = ["CE", "BCE"]
    lrs = [0.01, 0.005]
    weight_decays = [0.0, 5e-4]

    base_args = {'dataset': dataset,'device': device}

    for undirected, loss_function in product(undirecteds, loss_functions):
        additional_args = {'undirected': undirected,'loss_function': loss_function}

        if loss_function == "BCE":
            for predictor in predictors:
                if predictor == "InnerProduct":
                    param_combinations = product(lrs, weight_decays)
                    for lr, weight_decay in param_combinations:
                        args_dict = additional_args.copy()
                        args_dict.update({'outdim_predictor':1,'predictor': predictor,'lr': lr,'weight_decay': weight_decay})
                        cmd = build_cmd(base_args, args_dict)
                        #print(cmd)
                        subprocess.run(cmd, shell=True)
                else:
                    param_combinations = product(lrs, weight_decays, operators)
                    for lr, weight_decay, operator in param_combinations:
                        args_dict = additional_args.copy()
                        args_dict.update({'outdim_predictor':1,'predictor': predictor,'operator': operator,'lr': lr,'weight_decay': weight_decay})
                        cmd = build_cmd(base_args, args_dict)
                        #print(cmd)
                        subprocess.run(cmd, shell=True)
        else:  # loss_function == "CE"
            param_combinations = product(lrs, weight_decays)
            for lr, weight_decay in param_combinations:
                args_dict = additional_args.copy()
                args_dict.update({'outdim_predictor':2,'lr': lr,'weight_decay': weight_decay})
                cmd = build_cmd(base_args, args_dict)
                #print(cmd)
                subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()
