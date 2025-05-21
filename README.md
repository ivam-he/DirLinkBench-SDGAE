# DirLinkBench

This repository contains the implementation of our submitted paper **"Rethinking Link Prediction for Directed Graphs"**.

## Environment Setup
Ensure your environment meets the following dependencies:

- Python 3.10.13
- PyTorch 2.0.1
- PyTorch-CUDA 11.7
- torch-geometric 2.4.0
- NumPy 1.26.3
- SciPy 1.11.4
- scikit-learn 1.3.0
- ogb 1.3.6
- pandas 2.1.4
- DGL 0.9.0 (only required for DUPLEX)
- TensorBoard 2.10.1 (only required for ELTRA)
- TensorFlow 2.10.0 (only required for ELTRA)

## Datasets
This repository includes two small datasets: **Cora-ML** and **CiteSeer**. Additional datasets can be downloaded automatically when needed.

## Running Experiments
### Baselines
To run baselines such as **MLP, GCN, GAT, APPNP, GPRGNN, DGCN, DiGCN, DiGCNIB, DirGNN, MagNet, DUPLEX, DHYPR,** and **DiGAE**, use the following command:

```sh
python ./scripts/name.py --dataset data_name
```

- Replace `name.py` with the script name (e.g., `gcn.py`, `dirgnn.py`).
- Replace `data_name` with the dataset name (e.g., `cora_ml`, `citeseer`).
- This command automatically searches all parameters for baselines and stores the results in the `./results` folder.

### STRAP, ELTRA, and ODIN
These methods require generating embeddings before running the link prediction task.

#### ODIN
```sh
python -m odin.main_embedding --dataset data_name
python -m odin.main_lp --dataset data_name --predictor Logistic --operator cat
```

#### ELTRA
```sh
python -m eltra.main_embedding --dataset data_name
python -m eltra.main_lp --dataset data_name --predictor Logistic --operator cat
```

#### STRAP
STRAP requires converting the graph data into `.txt` format before generating embeddings using the official [STRAP repository](https://github.com/yinyuan1227/STRAP-git?tab=readme-ov-file):

```sh
python -m strap.data_transform --dataset data_name
python -m strap.main_lp --dataset data_name --predictor Logistic --operator cat
```

- `--predictor` options: `Logistic`, `InnerProduct`
- `--operator` options: `cat`, `hadamard`, `cat2`

### Our SDGAE
To search for optimal parameters for **SDGAE**, run:
```sh
python ./scripts/sdgae.py --dataset data_name
```
To reproduce directly the results in **Table 4**, run:
```sh
bash ./scripts/sdgae.sh
```


