python train_subset.py --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 --batch-size 128 --save_path ./checkpoint/pruned-dataset/pr_0.3 --subset_rate 0.3 --mask_path ./checkpoint/generated_mask/data_mask_
win10_ep30.npy --score_path ./checkpoint/generated_mask/score_win10_ep30.npy

python train_subset.py --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 --batch-size 128 --save_path ./checkpoint/pruned-dataset/pr_0.5 --subset_rate 0.5 --mask_path ./checkpoint/generated_mask/data_mask_
win10_ep30.npy --score_path ./checkpoint/generated_mask/score_win10_ep30.npy

python train_subset.py --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 --batch-size 128 --save_path ./checkpoint/pruned-dataset/pr_0.7 --subset_rate 0.7 --mask_path ./checkpoint/generated_mask/data_mask_
win10_ep30.npy --score_path ./checkpoint/generated_mask/score_win10_ep30.npy

python train_subset.py --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 --batch-size 64 --save_path ./checkpoint/pruned-dataset/pr_0.8 --subset_rate 0.8 --mask_path ./checkpoint/generated_mask/data_mask_w
in10_ep30.npy --score_path ./checkpoint/generated_mask/score_win10_ep30.npy

python train_subset.py --data_path ./data --dataset cifar10 --arch resnet18 --epochs 200 --learning_rate 0.1 --batch-size 32 --save_path ./checkpoint/pruned-dataset/pr_0.9 --subset_rate 0.9 --mask_path ./checkpoint/generated_mask/data_mask_w
in10_ep30.npy --score_path ./checkpoint/generated_mask/score_win10_ep30.npy