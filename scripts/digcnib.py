import argparse
from itertools import product
import subprocess

def build_cmd(base_args, additional_args):
    args = base_args.copy()
    args.update(additional_args)
    cmd_parts = ["python main_ddigcn.py", 
                 "--net DiGCNIB",
                 "--origin_feat True",
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

    predictors = ["MLPScore", "InnerProduct"]
    operators = ["cat", "hadamard"]
    loss_functions = ["CE", "BCE"]
    lrs = [0.01, 0.005]
    weight_decays = [0.0, 5e-4]
    alphas = [0.1, 0.2]

    base_args = {'dataset': dataset,'device': device}

    for loss_function in loss_functions:
        additional_args = {'loss_function': loss_function}
        if loss_function == "BCE":
            for predictor in predictors:
                if predictor == "InnerProduct":
                    param_combinations = product(lrs, weight_decays, alphas)
                    for lr, weight_decay, alpha in param_combinations:
                        args_dict = additional_args.copy()
                        args_dict.update({'outdim_predictor':1,'predictor': predictor,'lr': lr,'weight_decay': weight_decay, 'alpha':alpha})
                        cmd = build_cmd(base_args, args_dict)
                        #print(cmd)
                        subprocess.run(cmd, shell=True)
                else:
                    param_combinations = product(lrs, weight_decays, operators, alphas)
                    for lr, weight_decay, operator, alpha in param_combinations:
                        args_dict = additional_args.copy()
                        args_dict.update({'outdim_predictor':1,'predictor': predictor,'operator': operator,'lr': lr,'weight_decay': weight_decay, 'alpha':alpha})
                        cmd = build_cmd(base_args, args_dict)
                        #print(cmd)
                        subprocess.run(cmd, shell=True)
        else:  # loss_function == "CE"
            param_combinations = product(lrs, weight_decays, alphas)
            for lr, weight_decay, alpha in param_combinations:
                args_dict = additional_args.copy()
                args_dict.update({'outdim_predictor':2,'lr': lr,'weight_decay': weight_decay, 'alpha':alpha})
                cmd = build_cmd(base_args, args_dict)
                #print(cmd)
                subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()
