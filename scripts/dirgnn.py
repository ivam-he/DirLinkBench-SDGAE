import argparse
from itertools import product
import subprocess

def build_cmd(base_args, additional_args):
    args = base_args.copy()
    args.update(additional_args)
    cmd_parts = ["python main.py", 
                 "--net DirGNN",
                 "--origin_feat True",
                 "--embedding single",
                 "--save_res True",
                 f"--dataset {args.pop('dataset')}",
                 f"--device {args.pop('device')}"]
    
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                value = 'True'
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
    loss_functions = ["BCE"] #["CE", "BCE"]
    
    lrs = [0.01, 0.005]
    weight_decays = [0.0, 5e-4]
    alphas = [0.0, 0.5, 1.0]
    jks = ['cat', 'max']
    normalizes = [True, False]

    base_args = {'dataset': dataset,'device': device}

    for lr, weight_decay, jk, normalize, loss_function in product(lrs, weight_decays, jks, normalizes, loss_functions):
        additional_args = {'lr': lr,
            'weight_decay': weight_decay,
            'jk': jk,
            'normalize': normalize,
            'loss_function': loss_function
        }

        if loss_function == "BCE":
            for predictor in predictors:
                if predictor == "InnerProduct":
                    args_dict = additional_args.copy()
                    args_dict.update({'outdim_predictor':1,'predictor': predictor})
                    cmd = build_cmd(base_args, args_dict)
                    #print(cmd)
                    subprocess.run(cmd, shell=True)
                      
                else:
                    for operator in operators:
                        args_dict = additional_args.copy()
                        args_dict.update({'outdim_predictor':1,'predictor': predictor,'operator': operator})
                        cmd = build_cmd(base_args, args_dict)
                        #print(cmd)
                        subprocess.run(cmd, shell=True)
        else:  # loss_function == "CE"
            args_dict = additional_args.copy()
            args_dict.update({'outdim_predictor':2})
            cmd = build_cmd(base_args, args_dict)
            #print(cmd)
            subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main()
