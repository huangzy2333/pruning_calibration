import argparse
import os, sys
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dataset
import torchvision.transforms as transforms

from models import resnet
from utils import AverageMeter
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from data_subset import load_cifar100_sub, load_cifar10_sub

import tqdm

parser = argparse.ArgumentParser(description='evaluate the calibration of trained model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', default='./data', help='data store path.')
parser.add_argument('--dataset', default='cifar10', help='cifar10 and cifar100 are supported.', choices=['cifar10','cifar100'])
parser.add_argument('-bs', '--batch_size', default=128, type=int, help='the batch size for evaluation.')
parser.add_argument('--workers', default=0, type=int, help='the number of workers while loading data.')
parser.add_argument('--pin_memo', action='store_true', default=False, help='open pin_memory option while loading dataset')
parser.add_argument('--random_seed', default=12345, type=int, help='the random seed to split calibration and testing sets in the original testing data.')

parser.add_argument('--arch', default='resnet18', help='the architecture of trained model.')
parser.add_argument('--checkpoint_path', default=None, help='the path of trained model checkpoint to be evaluated.')
parser.add_argument('--subset_rate', help='the pruning rate of training data of the current checkpoint.')

parser.add_argument('--n_bins', default=10, help='the number of bins to calculate ece')
# parser.add_argument('--temperature', default=1, help='the temperature value for post-hoc temperature scaling to calibrate.')

parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
# parser.add_argument('--save_path', default=None, help='the path to save evaluation log and reliability diagrams.')


######### for debug
args = parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

## full dataset要用最后一个checkpoint！！！
args.checkpoint_path = './cifar10-checkpoint/all-dataset/checkpoint.pth.tar'
args.subset_rate = 0
args.dataset = 'cifar10'
args.save_path = os.path.split(args.checkpoint_path)[0]
args.workers = 2

# class ECE(nn.Module):
#     def __init__(self, n_bins):
#         super(ECE, self).__init__()
#         bin_boundaries = torch.linspace(0,1, n_bins+1)
#         self.bin_lowers = bin_boundaries[:-1]
#         self.bin_uppers = bin_boundaries[1:]
#
#     def forward(self, logits, labels):
#         softmaxes = F.softmax(logits, dim = 1) # logits dim: B, num_classes
#         confidences, predictions = torch.max(softmaxes, dim = 1) # torch.max返回value和index
#         accuracies = predictions == labels
#         ece = torch.zeros(1, device=logits.device)
#         for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
#             in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper) #bool value用*相当于&
#             bin_proportion = torch.mean(in_bin.float())
#             if bin_proportion >0:
#                 accuracy_in_bin = torch.mean(accuracies[in_bin].float())
#                 avg_conf_in_bin = torch.mean(confidences[in_bin].float())
#
#                 ece += torch.abs(avg_conf_in_bin - accuracy_in_bin) * bin_proportion
#         return ece

def compute_ece(logits, labels, n_bins=10, bin_conf_for_hb=None):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmaxes = F.softmax(logits, dim=1)  # logits dim: B, num_classes
    confidences, predictions = torch.max(softmaxes, dim=1)  # torch.max返回value和index
    if bin_conf_for_hb is not None:
        scaled_n_bins = np.linspace(0,1, len(bin_conf_for_hb)+1)
        conf_idx = np.digitize(confidences.cpu(), scaled_n_bins) - 1
        print(np.max(conf_idx))
        print(np.min(conf_idx))
        confidences = bin_conf_for_hb[conf_idx]
    accuracies = predictions == labels
    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)  # bool value用*相当于&
        bin_proportion = torch.mean(in_bin.float())
        # print(f"{bin_lower}-{bin_upper}: {torch.sum(in_bin.float())}%")
        if bin_proportion > 0:
            accuracy_in_bin = torch.mean(accuracies[in_bin].float())
            avg_conf_in_bin = torch.mean(confidences[in_bin].float())

            ece += torch.abs(avg_conf_in_bin - accuracy_in_bin) * bin_proportion
    return ece

def temp_scaling(logits, temp):
    temperature = temp.unsqueeze(1).expand(logits.size(0), logits.size(1)).to(logits.device)
    return logits / temperature

def search_temp(model, val_loader, device='cpu'):
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data = data.to(device)
            target = target.to(device)
            logits = model(data)
            if i == 0:
                all_logits = logits
                all_target = target
            else:
                all_logits = torch.concat((all_logits, logits), dim=0)
                all_target = torch.concat((all_target, target), dim=0)
        best_ece = 1000
        for t in range(0,1500,1):
            temp_scaled_logits = temp_scaling(all_logits,torch.ones(1)*t/100)
            ece_aft_temp_scaling = compute_ece(temp_scaled_logits, all_target)
            if ece_aft_temp_scaling < best_ece and ece_aft_temp_scaling != 0:
                # ece != 0是为了排除t=0时，scaled logits为infinity的情况，这时候由于存在不可计算的数，所以ece算出来是0
                best_temp = torch.ones(1).to(device)*t/100
                best_ece = ece_aft_temp_scaling
        return best_temp, best_ece

def hist_binning(logits, labels, n_bins):
    # binning partition on training set
    # get average accuracy in each bin on training set
    # set new confidence according to the originial confidence of data samples in validation and testing set
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmaxes = F.softmax(logits, dim=1)  # logits dim: B, num_classes
    confidences, predictions = torch.max(softmaxes, dim=1)  # torch.max返回value和index
    accuracies = predictions == labels
    bin_accuracies = torch.zeros(n_bins, device=logits.device)
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)  # bool value用*相当于&
        bin_proportion = torch.mean(in_bin.float())
        if bin_proportion > 0:
            accuracy_in_bin = torch.mean(accuracies[in_bin].float())
            bin_accuracies[i] = accuracy_in_bin
    return bin_accuracies

def search_nbins(model, val_loader, train_loader, device, eval_n_bins=10):
    best_n_bins = 0
    best_bin_confidences = torch.Tensor()
    best_ece = 1000

    model.eval()
    with torch.no_grad():
        # load training data
        for i, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target[0].to(device)
            logits = model(data)
            if i == 0:
                training_logits = logits
                training_target = target
            else:
                training_logits = torch.concat((training_logits, logits), dim=0)
                training_target = torch.concat((training_target, target), dim=0)

        # load val data
        for i, (data, target) in enumerate(val_loader):
            data = data.to(device)
            target = target.to(device)
            logits = model(data)
            if i == 0:
                all_logits = logits
                all_target = target
            else:
                all_logits = torch.concat((all_logits, logits), dim=0)
                all_target = torch.concat((all_target, target), dim=0)
        softmaxes = F.softmax(all_logits, dim=1)  # logits dim: B, num_classes
        confidences, predictions = torch.max(softmaxes, dim=1)  # torch.max返回value和index

    for n_bins in range(10,21,1):
        # compute scaled confidence in each bin on training data
        bin_accuracies = hist_binning(training_logits, training_target, n_bins)

        # compute ece on validation data after applying the hb
        # apply hb
        bin_boundaries = np.linspace(0,1,n_bins+1)
        conf_bin_idx = np.digitize(confidences.cpu(), bin_boundaries) - 1
        scaled_confidences = torch.tensor(bin_accuracies[conf_bin_idx], device=device)

        # evaluate ece on validation set
        accuracies = predictions == all_target
        bin_lowers = torch.linspace(0,1, eval_n_bins + 1)[:-1]
        bin_uppers = torch.linspace(0,1, eval_n_bins + 1)[1:]
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)  # bool value用*相当于&
            bin_proportion = torch.mean(in_bin.float())
            # print(f"{bin_lower}-{bin_upper}: {torch.sum(in_bin.float())}%")
            if bin_proportion > 0:
                accuracy_in_bin = torch.mean(accuracies[in_bin].float())
                avg_conf_in_bin = torch.mean(scaled_confidences[in_bin].float()) # use scaled conf

                ece += torch.abs(avg_conf_in_bin - accuracy_in_bin) * bin_proportion
        if ece < best_ece:
            best_n_bins = n_bins
            best_bin_confidences = bin_accuracies
            best_ece = ece
    return best_n_bins, best_bin_confidences, best_ece

def plot_reliability_diagrams(logits, labels, args):
    save_path = os.path.join(args.save_path, 'reliability diagram.png')
    n_bins = args.n_bins
    softmaxes = F.softmax(logits, dim=1)  # logits dim: B, num_classes
    confidences, predictions = torch.max(softmaxes, dim=1)  # torch.max返回value和index
    accuracies = torch.eq(predictions, labels)

    # Reliability diagram
    bins = torch.linspace(0, 1, n_bins + 1)
    bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]
    bin_accuracy = np.zeros(len(bin_indices),dtype=np.float64)
    bin_avg_conf = np.zeros(len(bin_indices), dtype=np.float64)
    ece = 0
    for i, bin_index in enumerate(bin_indices):
        if torch.mean(bin_index.float()) > 0:
            bin_accuracy[i] = torch.mean(accuracies[bin_index].float())
            bin_avg_conf[i] = torch.mean(confidences[bin_index].float())
            ece += torch.mean(bin_index.float()) * torch.abs(torch.mean(accuracies[bin_index].float()) - torch.mean(confidences[bin_index].float()))
        else:
            continue

    # plot based on bin widths, bin_accuracy, bin_avg_conf
    width = bins[1] - bins[0]

    #size and axis limits
    plt.figure(figsize=(6,6))
    plt.xlim(0,1)
    plt.ylim(0,1)

    #plot grid
    plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)

    #plot accuracy & confidence barplot
    plt.bar(bins[:-1], bin_accuracy, color='b', alpha=0.8, width=width, edgecolor = 'b', label='Outputs')
    gaps = bin_avg_conf - bin_accuracy
    plt.bar(bins[:-1], gaps, bottom=bin_accuracy, color='red', alpha=0.5, width=width, hatch='//', edgecolor='r', label='Gap')

    #plot y=x line
    plt.plot([0,1],[0,1], linestyle='--',color='grey')

    # label & legend
    plt.ylabel('Accuracy',fontsize=13)
    plt.xlabel('Confidence',fontsize=13)
    plt.legend(loc='upper left',framealpha=1.0)
    plt.text(0.95,0.05, f"ECE: {ece*100:.2f}%", transform=plt.gca().transAxes, fontsize=13,
             verticalalignment='bottom', horizontalalignment='right',
             bbox={'boxstyle': 'square,pad=0.3', 'edgecolor': 'black','facecolor': 'white'})
    if 'prune' in args.save_path:
        pruning_rate = args.save_path.split('/')[-2].split('_')[-1]
        title = f"{args.dataset} pruned data (pruning rate {pruning_rate})"
    else:
        title = f"{args.dataset} full data"
    plt.title(title, fontsize=15)

    plt.tight_layout()
    plt.savefig(save_path)

    return ece

def plot_conf_dist(logits, labels, args):
    # TO DO: plot the confidence distribution of testing set samples
    softmaxes = F.softmax(logits, dim=1)  # logits dim: B, num_classes
    confidences, predictions = torch.max(softmaxes, dim=1)  # torch.max返回value和index
    accuracies = predictions == labels
    errors = predictions != labels
    plt.figure(figsize=(6, 6))
    plt.xlim(0,1)

    print(torch.max(confidences))
    print(torch.min(confidences))
    plt.hist(confidences, bins=10, facecolor='blue', alpha=0.5, density=False, cumulative=False)
    # plt.hist(confidences[errors], bins=10, facecolor='red', alpha=0.5, density=False, cumulative=False)
    # plt.hist(confidences[accuracies], bins=10, facecolor='green', alpha=0.5, density=False, cumulative=False)



    # plt.hist(confidences[accuracies].float(), bins=np.linspace(0,1,21), facecolor='green', alpha=0.5, density=True)
    # plt.hist(confidences[errors].float(), bins=np.linspace(0,1,21), facecolor='red', alpha=0.5, density=True)
    plt.savefig(os.path.join(args.save_path,'confidence_distribution.png'))

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# def evaluate(test_loader, args, model, criterion, log): # criterion to calculate loss, no need here
def evaluate(train_loader, test_loader, val_loader, args, model, log):
    # losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():

        for i, (input, target) in enumerate(test_loader):
            if args.use_cuda:
                y = target.cuda()
                x = input.cuda()
            else:
                y = target
                x = input
            # compute output
            output = model(x)
            if not args.use_cuda:
                output = output.cpu()
            # loss = criterion(output, y)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, y, topk=(1, 5))
            # losses.update(loss.item(), len(y))
            top1.update(prec1.item(), len(y))
            top5.update(prec5.item(), len(y))
            if i == 0:
                all_logits = output
                all_target = y
            else:
                all_logits = torch.concat((all_logits, output), dim=0)
                all_target = torch.concat((all_target, y), dim=0)

        ece = compute_ece(all_logits, all_target, args.n_bins)
        # TO DO: add post-hoc calibration methods and re-evaluate
        best_t, _ = search_temp(model, val_loader, all_logits.device)
        scaled_logits = temp_scaling(all_logits, best_t)
        ece_ts = compute_ece(scaled_logits, all_target, args.n_bins)
        best_n_bins, bin_confidences, _ = search_nbins(model,val_loader, train_loader, all_logits.device) # 这里输入的logits应该是training dataset的logits
        # ece_hb = compute_ece(all_logits, all_target, args.n_bins, bin_conf_for_hb=bin_confidences)
        plot_reliability_diagrams(all_logits, all_target, args)
        plot_conf_dist(all_logits.cpu(), all_target.cpu(), args)

        print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}; ECE {ece:.3f}'.format(top1=top1, top5=top5,
                                                                                                    error1=100 - top1.avg, ece=ece.item()),
                log)
        print_log(f"**Post-hoc Temperature Scaling (grid search range temp 0-15)** ECE after TS: {ece_ts.item():.3f}; Best temp： {best_t}.", log)
        # print_log(f"**Post-hoc Histogram Binning (grid search range n_bins 10-20)** ECE after HB: {ece_hb.item():.3f}; Best n_bins： {best_n_bins}.", log)

        # make_model_diagrams(all_logits, all_target, n_bins=args.n_bins)

    # return top1.avg, losses.avg, ece
    return top1.avg, ece.item()

def main():
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    log = open(os.path.join(args.save_path, 'calibration_evaluation.txt'), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Dataset: {}".format(args.dataset), log)
    print_log("Data Path: {}".format(args.data_path), log)
    print_log(f"Random seed (for calibration & testing set split): {args.random_seed}.", log)
    print_log("Network: {}".format(args.arch), log)
    print_log("Batchsize: {}".format(args.batch_size), log)
    print_log(f"N_bins (the number of bins to calculate ECE): {args.n_bins}.", log)
    # print_log(f"Temperature (scaling for post-hoc calibration): {args.temperature}.", log)


    if args.dataset == 'cifar10':
        args.num_classes = 10
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        args.num_classes = 10
        args.num_samples = 50000 * (1 - args.subset_rate)
        data_mask = np.load('./cifar10-checkpoint/generated_mask/data_mask_win10_ep30.npy')
        sorted_score = np.load('./cifar10-checkpoint/generated_mask/score_win10_ep30.npy')
        train_loader, _ = load_cifar10_sub(args, data_mask, sorted_score)

        all_test_data = dataset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        indices = np.random.RandomState(args.random_seed).permutation(len(all_test_data.targets))  # data.targets就是cifar10里的labels
        calibration_set_indices = indices[:1000]
        test_set_indices = indices[1000:]
        calibration_data = torch.utils.data.Subset(all_test_data,calibration_set_indices)
        test_data = torch.utils.data.Subset(all_test_data, test_set_indices)

        test_loader = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=False,
                                                  num_workers=args.workers, pin_memory=args.pin_memo)
        calibration_loader = torch.utils.data.DataLoader(calibration_data, args.batch_size, shuffle=False,
                                                         num_workers=args.workers, pin_memory=args.pin_memo)
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        args.num_classes = 100
        args.num_samples = 50000 * (1 - args.subset_rate)
        data_mask = np.load('./cifar100-checkpoint/generated_mask/data_mask_win10_ep30.npy')
        sorted_score = np.load('./cifar100-checkpoint/generated_mask/score_win10_ep30.npy')

        train_loader, _ = load_cifar100_sub(args, data_mask, sorted_score)

        all_test_data = dataset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        indices = np.random.RandomState(args.random_seed).permutation(len(all_test_data.targets))  # data.targets就是cifar10里的labels
        calibration_set_indices = indices[:1000]
        test_set_indices = indices[1000:]
        calibration_data = torch.utils.data.Subset(all_test_data, calibration_set_indices)
        test_data = torch.utils.data.Subset(all_test_data, test_set_indices)

        test_loader = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=False,
                                                  num_workers=args.workers, pin_memory=args.pin_memo)
        calibration_loader = torch.utils.data.DataLoader(calibration_data, args.batch_size, shuffle=False,
                                                         num_workers=args.workers, pin_memory=args.pin_memo)

    model = torch.load(args.checkpoint_path)['state_dict']
    evaluate(train_loader, test_loader, calibration_loader, args, model, log)

if __name__ == '__main__':
    main()
    for eval_dataset in ['cifar100']:
        print("-"*20)
        print(f"Evaluating on {eval_dataset}...")
        for seed in ['77', '374', '565', '886', '4233']:
            for pruning_rate in tqdm.tqdm([0.3, 0.5, 0.7, 0.8, 0.9], desc=f"Evaluating run with seed {seed}"):
                args.checkpoint_path = './'+eval_dataset+'-checkpoint/pruned-dataset/pr_' + str(
                    pruning_rate) + '/seed' + seed + '/model_best.pth.tar'
                args.subset_rate = pruning_rate
                args.save_path = os.path.split(args.checkpoint_path)[0]
                args.dataset = eval_dataset
                main()