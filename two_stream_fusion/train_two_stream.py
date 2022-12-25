from mastoid_dataset import MastoidTwoSteamDataset, BatchSampler
from two_stream_model import TwoStreamFusion
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler, SGD

import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

classes = ["Tegmen", "SS", "EAC", "Open_antrum", "Facial_recess"]

def train_one_epoch(
        epoch_index, dataloader, model, loss_fn, optimizer, tb_writer, args):
    print("train")
    model.train()
    optimizer.zero_grad()
    
    running_loss, last_loss, running_step_acc, last_acc = 0., 0., 0., 0.
    running_step_acc_per_class = np.zeros(5)

    step_loss = 0.0
    step, prev_logging_step = 0, 0

    step_correct_preds, step_num_samples = 0, 0
    step_correct_preds_per_class = np.zeros(5)
    step_num_samples_per_class = np.zeros(5)
    for batch_idx, data in enumerate(tqdm(dataloader)):
        rgb, flow, label = data

        rgb = Variable(rgb.cuda())
        flow = Variable(flow.cuda())
        label = Variable(label.cuda())

        out = model((rgb, flow))
        preds = np.argmax(out.cpu().detach().numpy(), axis=1)
        gt = np.argmax(label.cpu().detach().numpy(), axis=1)

        cmp = preds == gt
        step_correct_preds += np.sum(cmp)
        step_correct_preds_per_class += np.bincount(gt[cmp], minlength=len(classes))
        step_num_samples += out.shape[0]
        step_num_samples_per_class += np.bincount(gt, minlength=len(classes))

        loss = loss_fn(out, label)
        loss = loss / args.accum_iter
        loss.backward()

        step_loss += loss.float()

        # One step
        if ((batch_idx + 1) % args.accum_iter == 0) or (batch_idx + 1 == len(dataloader)):
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            running_loss += step_loss
            step_loss = 0.

            step_acc = step_correct_preds / step_num_samples
            step_correct_preds, step_num_samples = 0, 0
            running_step_acc += step_acc

            step_acc_per_class = step_correct_preds_per_class / step_num_samples_per_class
            step_correct_preds_per_class = np.zeros(5)
            step_num_samples_per_class = np.zeros(5)
            running_step_acc_per_class += step_acc_per_class

            if (step % args.logging_per_steps == 0) or (batch_idx + 1 == len(dataloader)):
                num_steps = step - prev_logging_step
                prev_logging_step = step

                # Avg Loss
                last_loss = running_loss / num_steps
                running_loss = 0.
                # Avg acc
                last_acc = running_step_acc / num_steps
                running_step_acc = 0.0
                # Avg acc per class
                last_acc_per_class = running_step_acc_per_class / num_steps
                running_step_acc_per_class = np.zeros(5)

                # Logging.
                print(
                    f'\tstep {step - 1} loss {last_loss:.06f} acc {last_acc:.06f}')
                acc_per_class_str = '\tAcc per class: '
                for i in range(len(classes)):
                    acc_per_class_str += f'({classes[i]}, {last_acc_per_class[i]:06f}) '
                print(acc_per_class_str)

                # Write to tensor board.
                tb_x = epoch_index * len(dataloader) + batch_idx + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                tb_writer.add_scalar('Acc/train', last_acc, tb_x)
                for i in range(len(classes)):
                    tb_writer.add_scalar(f'Acc/train_{classes[i]}', last_acc_per_class[i], tb_x)

            # if (step % args.ckpt_per_steps == 0) or (batch_idx + 1 == len(dataloader)):
                # path = os.path.join(args.ckpt_path, f'ckpt_{step}.ckpt')
                # torch.save(model.module.state_dict(), path)
        path = os.path.join(args.ckpt_path, f'ckpt_epoch_{epoch_index}.ckpt')
        torch.save(model.module.state_dict(), path)
    return last_loss, last_acc


def val_one_epoch(dataloader, model, loss_fn):
    print("validation")
    model.eval()
    running_vloss = 0.0
    correct_preds, num_samples = 0, 0
    correct_preds_per_class = np.zeros(5)
    num_samples_per_class = np.zeros(5)
    for batch_idx, vdata in enumerate(tqdm(dataloader)):
        rgb, flow, label = vdata

        rgb = Variable(rgb.cuda())
        flow = Variable(flow.cuda())
        label = Variable(label.cuda())

        vout = model((rgb, flow))
        preds = np.argmax(vout.cpu().detach().numpy(), axis=1)
        gt = np.argmax(label.cpu().detach().numpy(), axis=1)
        
        cmp = preds == gt
        correct_preds += np.sum(cmp)
        correct_preds_per_class += np.bincount(gt[cmp], minlength=len(classes))
        num_samples += vout.shape[0]
        num_samples_per_class += np.bincount(gt, minlength=len(classes))

        vloss = loss_fn(vout, label)
        running_vloss += vloss.float()
    
    avg_vloss = running_vloss / (batch_idx + 1)
    acc = correct_preds / num_samples
    acc_per_class = correct_preds_per_class / num_samples_per_class
    
    acc_per_class_str = '\tAcc per class: '
    for i in range(len(classes)):
        acc_per_class_str += f'({classes[i]}, {acc_per_class[i]:06f}) '
    print(acc_per_class_str)
    
    return avg_vloss, acc


def train(dataloaders, model, loss_fn, optimizer, tb_writer, args):
    with torch.no_grad():
        avg_vloss, acc = val_one_epoch(dataloaders['val'], model, loss_fn)
    print(f'Initial validation loss {avg_vloss:06f} acc {acc:06f}')

    for epoch_idx in range(args.num_epochs):
        print(f'epoch {epoch_idx}')
        print('-' * 20)

        last_loss, last_acc = train_one_epoch(
            epoch_idx, dataloaders['train'],
            model, loss_fn, optimizer, tb_writer, args)

        with torch.no_grad():
            avg_vloss, acc = val_one_epoch(dataloaders['val'], model, loss_fn)

        print(f'LOSS train {last_loss:06f} valid {avg_vloss:06f}')
        tb_writer.add_scalars('Training vs. Validation Loss',
                              {'Training': last_loss, 'Validation': avg_vloss},
                              epoch_idx + 1)
        print(f'ACC train {last_acc:06f} valid {acc:06f}')
        tb_writer.add_scalars('Training vs. Validation ACC',
                              {'Training': last_acc, 'Validation': acc},
                              epoch_idx + 1)
        tb_writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="two_stream")
    parser.add_argument('-r', '--root', type=str, default="runs")
    parser.add_argument('-ag', '--accum_iter', type=int, default=4)
    parser.add_argument('-lps', '--logging_per_steps', type=int, default=10)
    parser.add_argument('-cps', '--ckpt_per_steps', type=int, default=20)

    parser.add_argument('-tv', '--train_videos',
                        nargs='+', type=int, required=True)
    parser.add_argument('-vv', '--val_videos', nargs='+',
                        type=int, required=True)
    parser.add_argument('-rgb', '--rgb_frames', type=int, default=5)
    parser.add_argument('-flow', '--opf_frames', type=int, default=10)
    parser.add_argument('--fps', type=int, default=15)

    parser.add_argument('-e', '--num_epochs', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-w', '--num_worders', type=int, default=32)

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    args = parser.parse_args()

    dataset = MastoidTwoSteamDataset(
        'train', args.fps, args.train_videos, args.rgb_frames, args.opf_frames, "class")
    dataloader = DataLoader(dataset,
                            batch_sampler=BatchSampler(
                                dataset.group_idxes, args.batch_size),
                            num_workers=args.num_worders)

    val_dataset = MastoidTwoSteamDataset(
        'val', args.fps, args.val_videos, args.rgb_frames, args.opf_frames, "class")
    val_dataloader = DataLoader(
        val_dataset,
        batch_sampler=BatchSampler(
            val_dataset.group_idxes, args.batch_size, shuffle=False),
        num_workers=args.num_worders)

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    if args.model == 'debug':
        run_path = f'./runs/{args.model}'
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_path = f'./runs/{args.model}_{timestamp}'
    tb_path = os.path.join(run_path, 'tb')
    writer = SummaryWriter(tb_path)

    ckpt_path = os.path.join(run_path, 'ckpts')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    args.ckpt_path = ckpt_path

    model = TwoStreamFusion(args.rgb_frames, args.opf_frames, 225, 400)
    model.cuda()
    model = nn.DataParallel(model)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    # Decay LR by a factor of 0.1 every 10 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train(dataloaders, model, loss_fn, optimizer, writer, args)
