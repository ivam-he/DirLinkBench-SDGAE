
python main_sdgae.py --net SDGAE --origin_feat True --runs 10 --save_res True --dataset cora_ml  --lr 0.01 --weight_decay 0.0 --nlayer 1 --K 5 --decoder mlpscore --operator hadamard
python main_sdgae.py --net SDGAE --origin_feat True --runs 10 --save_res True --dataset citeseer  --lr 0.01 --weight_decay 0.0 --nlayer 1 --K 5 --decoder mlpscore --operator hadamard

python main_sdgae.py --net SDGAE --origin_feat True --runs 10 --save_res True --dataset photo  --lr 0.005 --weight_decay 0.0 --nlayer 2 --K 5 --decoder mlpscore --operator hadamard
python main_sdgae.py --net SDGAE --origin_feat True --runs 10 --save_res True --dataset computers  --lr 0.005 --weight_decay 0.0 --nlayer 2 --K 3 --decoder mlpscore --operator hadamard

python main_sdgae.py --net SDGAE --runs 10 --save_res True --dataset wikics  --lr 0.005 --weight_decay 0.0005 --nlayer 2 --K 5 --decoder mlpscore --operator hadamard
python main_sdgae.py --net SDGAE --runs 10 --save_res True --dataset slashdot  --lr 0.01 --weight_decay 0.0005 --nlayer 2 --K 5 --decoder mlpscore --operator hadamard
python main_sdgae.py --net SDGAE --runs 10 --save_res True --dataset epinions  --lr 0.005 --weight_decay 0.0 --nlayer 2 --K 5 --decoder mlpscore --operator hadamard