import argparse
from itertools import product
import subprocess

def build_cmd(base_args, additional_args):
    args = base_args.copy()
    args.update(additional_args)
    cmd_parts = ["python main_sdgae.py", 
                 "--net SDGAE",
                 "--origin_feat True",
                 "--runs 2",
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

    decoders = ["mlpscore", "inner"]
    operators = ["cat", "hadamard"]
    
    lrs = [0.01, 0.005]
    weight_decays = [0.0, 5e-4]
    nlayers = [1,2] #For MLP in X initialization
    Ks = [3,4,5]

    base_args = {'dataset': dataset,'device': device}

    for lr, weight_decay, nlayer, K in product(lrs, weight_decays, nlayers, Ks):
        additional_args = {'lr': lr,
            'weight_decay': weight_decay,
            'nlayer': nlayer,
            'K': K
        }

        for decoder in decoders:
            if decoder == "inner":
                args_dict = additional_args.copy()
                args_dict.update({'decoder':decoder})
                cmd = build_cmd(base_args, args_dict)
                #print(cmd)
                subprocess.run(cmd, shell=True)
                      
            else:
                for operator in operators:
                    args_dict = additional_args.copy()
                    args_dict.update({'decoder':decoder,'operator': operator})
                    cmd = build_cmd(base_args, args_dict)
                    #print(cmd)
                    subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main()