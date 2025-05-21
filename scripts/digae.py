import argparse
from itertools import product
import subprocess

def build_cmd(base_args, additional_args):
    args = base_args.copy()
    args.update(additional_args)
    cmd_parts = ["python main_digae.py", 
                 "--net DiGAE",
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
    
    alphas = [0.0,0.2,0.4,0.6,0.8]
    betas = [0.0,0.2,0.4,0.6,0.8]
    lrs = [0.01, 0.005]
    weight_decays = [0.0, 5e-4]
    single_layers = [True, False]

    base_args = {'dataset': dataset,'device': device}
    for lr, weight_decay, alpha, beta, single_layer in product(lrs, weight_decays, alphas, betas, single_layers):
        additional_args = {'lr': lr,'weight_decay': weight_decay,'alpha': alpha, 'beta': beta, 'single_layer': single_layer}
        cmd = build_cmd(base_args, additional_args)
        #print(cmd)
        subprocess.run(cmd, shell=True)



if __name__ == "__main__":
    main()
