import numpy as np
import torch
import torch.optim as optim

from dlshogi import serializers
from dlshogi.swa import SWA
from dlshogi.data_loader import DataLoader

import argparse
import random

import logging

import wandb
import cshogi
import cairosvg

import io
from PIL import Image

torch.autograd.set_detect_anomaly(False)

parser = argparse.ArgumentParser(description='Traning RL policy network using hcpe')
parser.add_argument('train_data', type=str, nargs='+', help='train data file')
parser.add_argument('test_data', type=str, help='test data file')
parser.add_argument('--batchsize', '-b', type=int, default=1024, help='Number of positions in each mini-batch')
parser.add_argument('--testbatchsize', type=int, default=640, help='Number of positions in each test mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--network', type=str, default='wideresnet10', choices=['wideresnet10', 'wideresnet15', 'senet10', 'resnet10_swish', 'resnet10_swish2', 'resnet10_leakyrelu', 'resnet20_swish', 'resnet8', 'resnet8_dropout', 'resnet8_bottleneck2', 'resnet16'], help='network type')
parser.add_argument('--model', type=str, default='model_rl_val_hcpe', help='model file name')
parser.add_argument('--state', type=str, default='state_rl_val_hcpe', help='state file name')
parser.add_argument('--initmodel', '-m', default='', help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='', help='Resume the optimization from snapshot')
parser.add_argument('--log', default=None, help='log file path')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weightdecay_rate', type=float, default=0.0001, help='weightdecay rate')
parser.add_argument('--clip_grad_max_norm', type=float, default=10.0, help='max norm of the gradients')
parser.add_argument('--beta', type=float, default=0.001, help='entropy regularization coeff')
parser.add_argument('--val_lambda', type=float, default=0.333, help='regularization factor')
parser.add_argument('--policy_coef', type=float, default=0.5, help='policy loss coefficient')
parser.add_argument('--tanh', action='store_true')
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
parser.add_argument('--eval_interval', type=int, default=1000, help='evaluation interval')
parser.add_argument('--use_swa', action='store_true')
parser.add_argument('--swa_freq', type=int, default=250)
parser.add_argument('--swa_n_avr', type=int, default=10)
parser.add_argument('--swa_lr', type=float)
parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
parser.add_argument('--project', default='test', help='wandb project name')
parser.add_argument('--run_id', type=str, default=None, help='wandb run id and name')

args = parser.parse_args()

wandb.init(project=args.project, id=args.run_id, name=args.run_id.split('.')[0])
wandb.config.update(args)

if args.network == 'wideresnet15':
    from dlshogi.policy_value_network_wideresnet15 import *
elif args.network == 'senet10':
    from dlshogi.policy_value_network_senet10 import *
elif args.network == 'resnet10_swish':
    from dlshogi.policy_value_network_resnet10_swish import *
elif args.network == 'resnet20_swish':
    from dlshogi.policy_value_network_resnet20_swish import *
elif args.network == 'resnet10_swish2':
    from dlshogi.policy_value_network_resnet10_swish2 import *
elif args.network == 'resnet10_leakyrelu':
    from dlshogi.policy_value_network_resnet10_leakyrelu import *
elif args.network == 'resnet8':
    from dlshogi.policy_value_network_resnet8 import *
elif args.network == 'resnet8_dropout':
    from dlshogi.policy_value_network_resnet8_dropout import *
elif args.network == 'resnet8_bottleneck2':
    from dlshogi.policy_value_network_resnet8_bottleneck2 import *
elif args.network == 'resnet16':
    from dlshogi.policy_value_network_resnet16 import *
else:
    from dlshogi.policy_value_network import *

logging.basicConfig(
    format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO,
    handlers=[logging.FileHandler(args.log), logging.StreamHandler()],
)
logging.info('batchsize={}'.format(args.batchsize))
logging.info('MomentumSGD(lr={})'.format(args.lr))
logging.info('WeightDecay(rate={})'.format(args.weightdecay_rate))
logging.info('entropy regularization coeff={}'.format(args.beta))
logging.info('val_lambda={}'.format(args.val_lambda))

if args.gpu >= 0:
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cpu")

model = PolicyValueNetwork()
model.to(device)

wandb.watch(model, log_freq=args.eval_interval)

base_optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weightdecay_rate, nesterov=True)
if args.use_swa:
    optimizer = SWA(base_optimizer, swa_start=args.swa_freq, swa_freq=args.swa_freq, swa_lr=args.swa_lr, swa_n_avr=args.swa_n_avr)
else:
    optimizer = base_optimizer
cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()
bce_with_logits_loss_noreduce = torch.nn.BCEWithLogitsLoss(reduction='none')
if args.tanh:
    tanh = torch.nn.Tanh()
    mse_loss = torch.nn.MSELoss()
    mse_loss_noreduce = torch.nn.MSELoss(reduction='none')
if args.use_amp:
    logging.info('use amp')
scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

# Init/Resume
if args.initmodel:
    logging.info('Load model from {}'.format(args.initmodel))
    serializers.load_npz(args.initmodel, model)
if args.resume:
    logging.info('Load optimizer state from {}'.format(args.resume))
    checkpoint = torch.load(args.resume, map_location=device)
    epoch = checkpoint['epoch']
    t = checkpoint['t']
    base_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if args.use_amp and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
else:
    epoch = 0
    t = 0

logging.debug('read teacher data')
train_data = DataLoader.load_files(args.train_data)
logging.debug('read test data')
logging.debug(args.test_data)
test_data = np.fromfile(args.test_data, dtype=HuffmanCodedPosAndEval)

logging.info('train position num = {}'.format(len(train_data)))
logging.info('test position num = {}'.format(len(test_data)))

train_dataloader = DataLoader(train_data, args.batchsize, device, shuffle=True)
test_dataloader = DataLoader(test_data, args.testbatchsize, device)

# for SWA bn_update
def hcpe_loader(data, batchsize):
    for x1, x2, t1, t2, value in DataLoader(data, batchsize, device):
        yield x1, x2

def accuracy(y, t):
    return (torch.max(y, 1)[1] == t).sum().item() / len(t)

def binary_accuracy(y, t):
    pred = y >= 0
    truth = t >= 0.5
    return pred.eq(truth).sum().item() / len(t)

def log_example(t, hcpevec, loss_policy, loss_result, loss_value, t1, t2, value, y1, y2):
    logging.debug(loss_policy.shape)
    # loss_policy
    test_data_at = wandb.Artifact("test_samples_" + wandb.run.id, type="predictions")
    test_table = wandb.Table(columns=["t", "label", "position", "turn", "t1", "y1", "t2", "value", "y2", "loss_policy", "loss_result", "loss_value", "sfen"])
    _log_example(loss_policy, 'good loss_policy', test_table, t, hcpevec, loss_policy, loss_result, loss_value, t1, t2, value, y1, y2)
    _log_example(-loss_policy, 'bad loss_policy', test_table, t, hcpevec, loss_policy, loss_result, loss_value, t1, t2, value, y1, y2)
    _log_example(loss_result, 'good loss_result', test_table, t, hcpevec, loss_policy, loss_result, loss_value, t1, t2, value, y1, y2)
    _log_example(-loss_result, 'bad loss_result', test_table, t, hcpevec, loss_policy, loss_result, loss_value, t1, t2, value, y1, y2)
    _log_example(loss_value, 'good loss_value', test_table, t, hcpevec, loss_policy, loss_result, loss_value, t1, t2, value, y1, y2)
    _log_example(-loss_value, 'bad loss_value', test_table, t, hcpevec, loss_policy, loss_result, loss_value, t1, t2, value, y1, y2)
    test_data_at.add(test_table, "predictions")
    wandb.run.log_artifact(test_data_at)

def _log_example(sortvec, label, table, t, hcpevec, loss_policy, loss_result, loss_value, t1, t2, value, y1, y2):
    sort_idx = np.argsort(sortvec.to('cpu').detach().numpy().copy())
    board = cshogi.Board()
    for i in range(5):
        j = sort_idx[i]
        png_file = io.BytesIO()
        board.set_hcp(hcpevec[j]['hcp'])
        svg = board.to_svg().replace('serif', 'Noto Serif CJK JP')
        cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to=png_file)
        image = wandb.Image(Image.open(png_file))
        y1_max = torch.argmax(y1[j]).item()
        if board.turn == 1:
            y1_max = 80 - y1_max
        table.add_data(t, label, image, board.turn, cshogi.move_to_usi(hcpevec[j]['bestMove16']), cshogi.SQUARE_NAMES[y1_max % 81], t2[j].item(), value[j].item(), y2[j].item(), loss_policy[j], loss_result[j], loss_value[j], board.sfen())
        png_file.close()


# train
itr = 0
sum_loss1 = 0
sum_loss2 = 0
sum_loss3 = 0
sum_loss = 0
eval_interval = args.eval_interval
softplus = nn.Softplus()
for e in range(args.epoch):
    itr_epoch = 0
    sum_loss1_epoch = 0
    sum_loss2_epoch = 0
    sum_loss3_epoch = 0
    sum_loss_epoch = 0
    for x1, x2, t1, t2, value in train_dataloader:
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            model.train()

            y1, y2 = model(x1, x2)
            # z = t2.view(-1) - value.view(-1) + 0.5
            # z = softplus(value.view(-1) - y2.view(-1)) * 0.01 + 1.
            z = t2.view(-1) + 0.5

            model.zero_grad()
            loss1 = (cross_entropy_loss(y1, t1) * z).mean()
            if args.beta > 0:
                loss1 += args.beta * (F.softmax(y1, dim=1) * F.log_softmax(y1, dim=1)).sum(dim=1).mean()
            if args.tanh:
                loss2 = mse_loss(tanh(y2), t2 * 2. - 1.)
                loss3 = mse_loss(tanh(y2), value * 2. - 1.)
            else:
                loss2 = bce_with_logits_loss(y2, t2)
                loss3 = bce_with_logits_loss(y2, value)
            loss = args.policy_coef * loss1 + (1. - args.policy_coef) * ((1 - args.val_lambda) * loss2 + args.val_lambda * loss3)

        scaler.scale(loss).backward()
        if args.clip_grad_max_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_max_norm)
        scaler.step(optimizer)
        scaler.update()

        t += 1
        itr += 1
        sum_loss1 += loss1.item()
        sum_loss2 += loss2.item()
        sum_loss3 += loss3.item()
        sum_loss += loss.item()
        itr_epoch += 1
        sum_loss1_epoch += loss1.item()
        sum_loss2_epoch += loss2.item()
        sum_loss3_epoch += loss3.item()
        sum_loss_epoch += loss.item()

        # print train loss
        if t % eval_interval == 0 or (t == 1):
            model.eval()

            x1, x2, t1, t2, value, hcpevec = test_dataloader.sample_test()
            with torch.no_grad():
                y1, y2 = model(x1, x2)

                loss1_noreduce = cross_entropy_loss(y1, t1)
                loss1 = loss1_noreduce.mean()
                if args.tanh:
                    loss2_noreduce = mse_loss_noreduce(tanh(y2), t2 * 2. - 1.)
                    loss3_noreduce = mse_loss_noreduce(tanh(y2), value * 2. - 1.)
                else:
                    loss2_noreduce = bce_with_logits_loss_noreduce(y2, t2)
                    loss3_noreduce = bce_with_logits_loss_noreduce(y2, value)
                loss2_noreduce = loss2_noreduce.flatten()
                loss3_noreduce = loss3_noreduce.flatten()
                logging.debug(f'x1: {x1.shape}')
                logging.debug(f'x2: {x2.shape}')
                logging.debug(f't1: {t1.shape}')
                logging.debug(f't2: {t2.shape}')
                logging.debug(f'y1: {y1.shape}')
                logging.debug(f'y2: {y2.shape}')
                logging.debug(f't1[0]: {t1[0]}')
                logging.debug(f'y1[0]: {y1[0]}')
                logging.debug(f'y1[0].sum(): {y1[0].sum()}')
                loss2 = loss2_noreduce.mean()
                loss3 = loss3_noreduce.mean()
                loss = loss1 + (1 - args.val_lambda) * loss2 + args.val_lambda * loss3

                logging.info('epoch = {}, iteration = {}, loss = {:.08f}, {:.08f}, {:.08f}, {:.08f}, test loss = {:.08f}, {:.08f}, {:.08f}, {:.08f}, test accuracy = {:.08f}, {:.08f}'.format(
                    epoch + 1, t,
                    sum_loss1 / itr, sum_loss2 / itr, sum_loss3 / itr, sum_loss / itr,
                    loss1.item(), loss2.item(), loss3.item(), loss.item(),
                    accuracy(y1, t1), binary_accuracy(y2, t2)))

                wandb.log({
                    "train/loss_policy": sum_loss1 / itr, "train/loss_result": sum_loss2 / itr, "train/loss_value": sum_loss3 / itr, "train/loss_sum": sum_loss / itr,
                    "valid/loss_policy": loss1.item(), "valid/loss_result": loss2.item(), "valid/loss_value": loss3.item(), "valid/loss_sum": loss.item(), "valid_acc/acc_policy": accuracy(y1, t1), "valid_acc/acc_result": binary_accuracy(y2,t2),
                }, step=t)
                log_example(t, hcpevec, loss1_noreduce, loss2_noreduce, loss3_noreduce, t1, t2, value, y1, y2)

            itr = 0
            sum_loss1 = 0
            sum_loss2 = 0
            sum_loss3 = 0
            sum_loss = 0

    if args.use_swa:
        optimizer.swap_swa_sgd()

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            optimizer.bn_update(hcpe_loader(train_data, args.batchsize), model)

    # print train loss for each epoch
    itr_test = 0
    sum_test_loss1 = 0
    sum_test_loss2 = 0
    sum_test_loss3 = 0
    sum_test_loss = 0
    sum_test_accuracy1 = 0
    sum_test_accuracy2 = 0
    sum_test_entropy1 = 0
    sum_test_entropy2 = 0
    model.eval()
    with torch.no_grad():
        for x1, x2, t1, t2, value in test_dataloader:
            y1, y2 = model(x1, x2)

            itr_test += 1
            loss1_noreduce = cross_entropy_loss(y1, t1)
            loss1 = loss1_noreduce.mean()
            if args.tanh:
                loss2_noreduce = mse_loss_noreduce(tanh(y2), t2 * 2. - 1.)
                loss3_noreduce = mse_loss_noreduce(tanh(y2), value * 2. - 1.)
            else:
                loss2_noreduce = bce_with_logits_loss_noreduce(y2, t2)
                loss3_noreduce = bce_with_logits_loss_noreduce(y2, value)
            loss2 = loss2_noreduce.mean()
            loss3 = loss3_noreduce.mean()

            loss = loss1 + (1 - args.val_lambda) * loss2 + args.val_lambda * loss3
            sum_test_loss1 += loss1.item()
            sum_test_loss2 += loss2.item()
            sum_test_loss3 += loss3.item()
            sum_test_loss += loss.item()
            sum_test_accuracy1 += accuracy(y1, t1)
            sum_test_accuracy2 += binary_accuracy(y2, t2)

            entropy1 = (- F.softmax(y1, dim=1) * F.log_softmax(y1, dim=1)).sum(dim=1)
            sum_test_entropy1 += entropy1.mean().item()

            p2 = y2.sigmoid()
            #entropy2 = -(p2 * F.log(p2) + (1 - p2) * F.log(1 - p2))
            log1p_ey2 = F.softplus(y2)
            entropy2 = -(p2 * (y2 - log1p_ey2) + (1 - p2) * -log1p_ey2)
            sum_test_entropy2 +=entropy2.mean().item()

        logging.info('epoch = {}, iteration = {}, train loss avr = {:.08f}, {:.08f}, {:.08f}, {:.08f}, test_loss = {:.08f}, {:.08f}, {:.08f}, {:.08f}, test accuracy = {:.08f}, {:.08f}, test entropy = {:.08f}, {:.08f}'.format(
            epoch + 1, t,
            sum_loss1_epoch / itr_epoch, sum_loss2_epoch / itr_epoch, sum_loss3_epoch / itr_epoch, sum_loss_epoch / itr_epoch,
            sum_test_loss1 / itr_test, sum_test_loss2 / itr_test, sum_test_loss3 / itr_test, sum_test_loss / itr_test,
            sum_test_accuracy1 / itr_test, sum_test_accuracy2 / itr_test,
            sum_test_entropy1 / itr_test, sum_test_entropy2 / itr_test))

    epoch += 1

    if args.use_swa:
        if e != args.epoch - 1:
            optimizer.swap_swa_sgd()

logging.info('save the model to {}'.format(args.model))
serializers.save_npz(args.model, model)
logging.info('save the optimizer to {}'.format(args.state))
state = {
    'epoch': epoch,
    't': t,
    'optimizer_state_dict': base_optimizer.state_dict(),
    }
if args.use_amp:
    state['scaler_state_dict'] = scaler.state_dict()
torch.save(state, args.state)
