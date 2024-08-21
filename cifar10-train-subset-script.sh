#for dataset in 'cifar10' 'cifar100'
#do
#  for seed in 37 42 77 88 374
#  do
#    for pruning_rate in 0.3 0.5 0.7
#    do
#      pass
#    done
#
#
#  done
#done

for dataset in cifar10 cifar100
do
  for seed in 565 42 77 886 374
  do
    for pruning_rate in 0.3 0.5 0.7
    do
      batch_size=128
      echo train_subset.py --data_path ./data --dataset $dataset --arch resnet18 --epochs 200 --learning_rate 0.1 --batch-size $batch_size --save_path ./$dataset-checkpoint/pruned-dataset/pr_$pruning_rate --subset_rate $pruning_rate --mask_path ./$dataset-checkpoint/generated_mask/data_mask_win10_ep30.npy --score_path ./$dataset-checkpoint/generated_mask/score_win10_ep30.npy --manualSeed $seed --workers 0
      python train_subset.py --data_path ./data --dataset $dataset --arch resnet18 --epochs 200 --learning_rate 0.1 --batch-size $batch_size --save_path ./$dataset-checkpoint/pruned-dataset/pr_$pruning_rate --subset_rate $pruning_rate --mask_path ./$dataset-checkpoint/generated_mask/data_mask_win10_ep30.npy --score_path ./$dataset-checkpoint/generated_mask/score_win10_ep30.npy --manualSeed $seed --workers 0

    done
    pruning_rate=0.8
    batch_size=64
    echo train_subset.py --data_path ./data --dataset $dataset --arch resnet18 --epochs 200 --learning_rate 0.1 --batch-size $batch_size --save_path ./$dataset-checkpoint/pruned-dataset/pr_$pruning_rate --subset_rate $pruning_rate --mask_path ./$dataset-checkpoint/generated_mask/data_mask_win10_ep30.npy --score_path ./$dataset-checkpoint/generated_mask/score_win10_ep30.npy --manualSeed $seed --workers 0
    python train_subset.py --data_path ./data --dataset $dataset --arch resnet18 --epochs 200 --learning_rate 0.1 --batch-size $batch_size --save_path ./$dataset-checkpoint/pruned-dataset/pr_$pruning_rate --subset_rate $pruning_rate --mask_path ./$dataset-checkpoint/generated_mask/data_mask_win10_ep30.npy --score_path ./$dataset-checkpoint/generated_mask/score_win10_ep30.npy --manualSeed $seed --workers 0

    pruning_rate=0.9
    batch_size=32
    echo train_subset.py --data_path ./data --dataset $dataset --arch resnet18 --epochs 200 --learning_rate 0.1 --batch-size $batch_size --save_path ./$dataset-checkpoint/pruned-dataset/pr_$pruning_rate --subset_rate $pruning_rate --mask_path ./$dataset-checkpoint/generated_mask/data_mask_win10_ep30.npy --score_path ./$dataset-checkpoint/generated_mask/score_win10_ep30.npy --manualSeed $seed --workers 0
    python train_subset.py --data_path ./data --dataset $dataset --arch resnet18 --epochs 200 --learning_rate 0.1 --batch-size $batch_size --save_path ./$dataset-checkpoint/pruned-dataset/pr_$pruning_rate --subset_rate $pruning_rate --mask_path ./$dataset-checkpoint/generated_mask/data_mask_win10_ep30.npy --score_path ./$dataset-checkpoint/generated_mask/score_win10_ep30.npy --manualSeed $seed --workers 0

  done
done