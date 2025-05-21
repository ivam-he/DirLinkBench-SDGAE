import argparse
from itertools import product
import subprocess

def build_cmd(base_args, additional_args):
    args = base_args.copy()
    args.update(additional_args)
    cmd_parts = ["python main.py", 
                 "--net MagNet",
                 "--origin_feat True",
                 "--embedding dual",
                 "--save_res True",
                 "--loss_function CE",
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
    
    lrs = [0.01, 0.005]
    weight_decays = [0.0, 5e-4]
    qs = [0.05, 0.1, 0.15, 0.2, 0.25]
    Ks = [1, 2]

    base_args = {'dataset': dataset,'device': device}
    for lr, weight_decay, q, K in product(lrs, weight_decays, qs, Ks):
        additional_args = {'lr': lr,'weight_decay': weight_decay,'q': q, 'K': K}
        cmd = build_cmd(base_args, additional_args)
        #print(cmd)
        subprocess.run(cmd, shell=True)

        


if __name__ == "__main__":
    main()
