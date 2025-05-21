import argparse
from itertools import product
import subprocess


def build_cmd(base_args, additional_args):
    args = base_args.copy()
    args.update(additional_args)
    cmd_parts = ["python main_duplex.py", 
                 "--net DUPLEX",
                 "--origin_feat True",
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

    loss_weights = [0.1,0.3]
    loss_decays = [0,1e-4,1e-2]
    lrs = [0.01, 0.001]
    weight_decays = [0.0, 5e-4]

    base_args = {'dataset': dataset,'device': device}
    for lr, weight_decay, loss_weight, loss_decay in product(lrs, weight_decays, loss_weights, loss_decays):
        additional_args = {'lr': lr,'weight_decay': weight_decay,'loss_weight': loss_weight, 'loss_decay': loss_decay}
        cmd = build_cmd(base_args, additional_args)
        #print(cmd)
        subprocess.run(cmd, shell=True)

        
if __name__ == "__main__":
    main()
