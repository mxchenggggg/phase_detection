from mastoid_dataset import MastoidTwoSteamDataset, BatchSampler, MastoidTransform
from two_stream_model import TwoStreamFusion
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler, SGD, Adam

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sn

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from torchmetrics import Accuracy, Precision, ConfusionMatrix

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'


def train_one_epoch(
        epoch_idx, dataloader, model, loss_fn, optimizer, scaler, tb_writer,
        args):
    print("train")
    model.train()
    optimizer.zero_grad()

    running_loss, last_loss = 0., 0.
    last_eval = None

    step_loss = 0.0
    step, prev_logging_step = 0, 0

    logging_target_list = []
    logging_preds_list = []

    skip_lr_scheduler = False

    for batch_idx, data in enumerate(tqdm(dataloader)):
        rgb, flow, label = data

        rgb = rgb.cuda()
        flow = flow.cuda()
        label = label.cuda()

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out = model((rgb, flow))
            loss = loss_fn(out, label)
            loss = loss / args.accum_iter

        scaler.scale(loss).backward()
        # loss.backward()

        step_loss += loss.float()

        logging_target_list.append(torch.argmax(label.cpu().detach(), axis=1))
        logging_preds_list.append(torch.argmax(out.cpu().detach(), axis=1))

        # One step
        if ((batch_idx + 1) % args.accum_iter == 0) or (batch_idx + 1 == len(dataloader)):
            scaler.step(optimizer)

            scale = scaler.get_scale()
            scaler.update()
            skip_lr_scheduler = (scale != scaler.get_scale())

            # optimizer.step()

            optimizer.zero_grad()

            step += 1
            running_loss += step_loss
            step_loss = 0.

            if (step % args.logging_per_steps == 0) or (batch_idx + 1 == len(dataloader)):
                # Avg Loss
                last_loss = running_loss / (step - prev_logging_step)
                prev_logging_step = step
                running_loss = 0.

                # Evaluation
                last_eval = report_metrics(
                    torch.concat(logging_target_list),
                    torch.concat(logging_preds_list),
                    logging=False)
                logging_target_list = []
                logging_preds_list = []

                acc = last_eval['metrics'].loc['ACC']['all']
                prec = last_eval['metrics'].loc['PREC']['all']
                # print(last_eval['metrics'])
                print(
                    f"\tstep {step - 1} loss {last_loss:.06f} acc {acc:.06f} prec {prec:.06f}")

                # Write to tensor board.
                tb_x = epoch_idx * len(dataloader) + batch_idx + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                tb_writer.add_scalar(
                    'Acc/train', last_eval['metrics'].loc['ACC']['all'], tb_x)
                tb_writer.add_scalar(
                    'Prec/train', last_eval['metrics'].loc['PREC']['all'], tb_x)

                for class_name in classes:
                    tb_writer.add_scalar(
                        f'Acc/train_{class_name}',
                        last_eval['metrics'].loc['ACC'][class_name], tb_x)
                    tb_writer.add_scalar(
                        f'Prec/train_{class_name}',
                        last_eval['metrics'].loc['PREC'][class_name], tb_x)

            # if (step % args.ckpt_per_steps == 0) or (batch_idx + 1 == len(dataloader)):
                # path = os.path.join(args.ckpt_path, f'ckpt_{step}.ckpt')
                # torch.save(model.module.state_dict(), path)
    return last_loss, last_eval, skip_lr_scheduler


def val_one_epoch(dataloader, model, loss_fn, args):
    print("validation")
    model.eval()
    running_vloss = 0.0

    target_list = []
    preds_list = []

    for batch_idx, vdata in enumerate(tqdm(dataloader)):
        rgb, flow, label = vdata

        rgb = rgb.cuda()
        flow = flow.cuda()
        label = label.cuda()

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            vout = model((rgb, flow))

        preds_list.append(torch.argmax(vout.cpu().detach(), axis=1))
        target_list.append(torch.argmax(label.cpu().detach(), axis=1))

        vloss = loss_fn(vout, label)
        running_vloss += vloss.float()

    avg_vloss = running_vloss / (batch_idx + 1)

    eval_result = report_metrics(
        torch.concat(target_list),
        torch.concat(preds_list),
        logging=True, name='validation result')

    return avg_vloss, eval_result


def train(
        start_epoch, dataloaders, model, loss_fn, optimizer, scheduler, scaler,
        tb_writer, args):
    with torch.no_grad():
        avg_vloss, eval_result = val_one_epoch(
            dataloaders['val'], model, loss_fn, args)
    print(f'Initial validation loss {avg_vloss:06f}')

    for epoch_idx in range(start_epoch, args.num_epochs):
        print(f'epoch {epoch_idx}')
        print('-' * 20)
        # Train.
        last_loss, last_train_eval, skip_lr_scheduler = train_one_epoch(
            epoch_idx, dataloaders['train'],
            model, loss_fn, optimizer, scaler, tb_writer, args)
        print("Last train eval")
        print(last_train_eval['metrics'])
        print(last_train_eval['cm'])
        print(last_train_eval['cm_norm'])

        # Validation.
        with torch.no_grad():
            avg_vloss, val_eval = val_one_epoch(
                dataloaders['val'],
                model, loss_fn, args)

        # Logging.
        print(f'LOSS train {last_loss:06f} valid {avg_vloss:06f}')
        tb_writer.add_scalars('Training vs. Validation Loss',
                              {'Training': last_loss, 'Validation': avg_vloss},
                              epoch_idx + 1)
        for m in ['ACC', 'PREC']:
            last_train_all = last_train_eval['metrics'].loc[m]['all']
            val_all = val_eval['metrics'].loc[m]['all']
            print(f'{m} train {last_train_all:06f} valid {val_all:06f}')
            tb_writer.add_scalars(
                f'Training vs. Validation {m}',
                {'Training': last_train_all, 'Validation': val_all},
                epoch_idx)

        tb_writer.add_figure(
            "CM/train", get_cm_plot(last_train_eval['cm_norm']),
            epoch_idx)
        tb_writer.add_figure(
            "CM/val", get_cm_plot(val_eval['cm_norm']),
            epoch_idx)

        tb_writer.flush()

        # if not skip_lr_scheduler:
        # scheduler.step()

        if (epoch_idx + 1) % args.ckpt_per_epoches == 0 or epoch_idx + 1 == args.num_epochs:
            path = os.path.join(
                args.ckpt_path, f'ckpt_epoch_{epoch_idx:02d}.ckpt')
            torch.save({'epoch': epoch_idx,
                        'model_state_dict': model.module.state_dict(),
                        'opt_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'args': args,
                        'val_eval': val_eval,
                        'last_train_eval': last_train_eval}, path)
            print(f'save checkpoint {path}')


if __name__ == "__main__":
    # --------- Arg Parse ---------

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="two_stream")
    parser.add_argument('-r', '--root', type=str, default="runs")
    parser.add_argument('-ag', '--accum_iter', type=int, default=2)
    parser.add_argument('-lps', '--logging_per_steps', type=int, default=10)
    parser.add_argument('-cpe', '--ckpt_per_epoches', type=int, default=5)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00001)
    parser.add_argument('-ckpt', '--checkpoint', type=str, default="")
    parser.add_argument('-no_aug', '--no_augmentation',
                        action='store_true', default=False)
    parser.add_argument('-f', '--freeze_base',
                        action='store_true', default=False)
    parser.add_argument('-dp', '--dropout', type=float, default=0.5)
    parser.add_argument('-e', '--num_epochs', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-amp', '--use_amp', action='store_true', default=True)
    parser.add_argument('-bb', '--backbone', type=str, default="vgg16")

    parser.add_argument('-tv', '--train_videos',
                        nargs='+', type=int)
    parser.add_argument('-vv', '--val_videos', nargs='+',
                        type=int)

    parser.add_argument('-rgb', '--rgb_frames', type=int, default=5)
    parser.add_argument('-flow', '--opf_frames', type=int, default=10)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('-w', '--num_workers', type=int, default=8)
    parser.add_argument('-cj', '--color_jitter_p', type=float, default=0.5)
    parser.add_argument('-cmode', '--class_mode', type=str, default='Step')

    args = parser.parse_args()
    
    print(args)
    
    checkpoint = None
    if args.checkpoint != "":
        print(f'Loaded from checkpoint {args.checkpoint}.')
        checkpoint = torch.load(args.checkpoint)
        new_num_epochs = args.num_epochs
        args = checkpoint['args']
        args.num_epochs = new_num_epochs
    else:
        if args.model == 'debug':
            run_path = f'./runs/{args.model}'
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_path = f'./runs/{args.model}_{timestamp}'
        args.tb_path = os.path.join(run_path, 'tb')

        ckpt_path = os.path.join(run_path, 'ckpts')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        args.ckpt_path = ckpt_path
    writer = SummaryWriter(args.tb_path)

    if args.class_mode == 'Task':
        classes = ["Tegmen", "SS", "EAC", "Open_antrum", "Facial_recess"]
        classes_loss_weights = [1.1, 1.1, 1.1, 1.0, 1.0]
    else:
        classes = ["Expose", "Antrum", "Facial_recess"]
        classes_loss_weights = [1.0, 1.05, 1.0]
    num_classes = len(classes)
    
    print('clsses: ', classes)
    print(f"Data augmentation: {'False' if args.no_augmentation else 'True'}")
    print(f"Freeze pretrained model: {'True' if args.freeze_base else 'False'}")
    print(f"Experiment: {args.model}")

    # --------- Evaluation ---------

    acc_all_m = Accuracy(num_classes=num_classes, average='macro')
    acc_per_m = Accuracy(num_classes=num_classes, average=None)

    prec_all_m = Precision(num_classes=num_classes, average='macro')
    prec_per_m = Precision(num_classes=num_classes, average=None)

    c_matrix_m = ConfusionMatrix(num_classes=num_classes)
    c_matrix_norm_m = ConfusionMatrix(num_classes=num_classes, normalize='true')

    def report_metrics(target, preds, logging=False, name=None):
        acc_all = acc_all_m(preds, target)
        acc_all_m.reset()
        acc_per = acc_per_m(preds, target)
        acc_per_m.reset()
        acc = np.append(acc_per, acc_all)

        prec_all = prec_all_m(preds, target)
        prec_all_m.reset()
        prec_per = prec_per_m(preds, target)
        prec_per_m.reset()
        prec = np.append(prec_per, prec_all)

        metrics_df = pd.DataFrame(
            np.stack([acc, prec]),
            index=['ACC', 'PREC'],
            columns=classes + ['all'])

        c_matrix = c_matrix_m(preds, target).numpy()
        c_matrix_m.reset()
        c_matrix_norm = c_matrix_norm_m(preds, target).numpy()
        c_matrix_norm_m.reset()

        cm_df = pd.DataFrame(c_matrix, index=classes, columns=classes)
        cm_df_norm = pd.DataFrame(c_matrix_norm, index=classes, columns=classes)

        if logging:
            print(name)
            print(metrics_df)
            print(cm_df)
            print(cm_df_norm)

        return {'metrics': metrics_df, 'cm': cm_df, 'cm_norm': cm_df_norm}

    def get_cm_plot(cm_df):
        plt.figure(figsize=(12, 7))
        return sn.heatmap(cm_df, annot=True).get_figure()

    # ----- Datasets Dataloader ----------

    use_train_aug = not args.no_augmentation
    train_transform = MastoidTransform(
        augment=use_train_aug, hflip_p=0.5, affine_p=0.5, rotate_angle=0.,
        scale_range=(0.9, 1.1),
        color_jitter_p=args.color_jitter_p, brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.1)

    dataset = MastoidTwoSteamDataset(
        'train', train_transform, args.fps, args.train_videos, args.rgb_frames,
        args.opf_frames, "video", args.class_mode)
    dataloader = DataLoader(dataset,
                            batch_sampler=BatchSampler(
                                dataset.group_idxes, args.batch_size),
                            num_workers=args.num_workers)

    val_transform = MastoidTransform(augment=False)

    val_dataset = MastoidTwoSteamDataset(
        'val', val_transform, args.fps, args.val_videos, args.rgb_frames, args.
        opf_frames, "video", args.class_mode)
    val_dataloader = DataLoader(
        val_dataset,
        batch_sampler=BatchSampler(
            val_dataset.group_idxes, args.batch_size, shuffle=False),
        num_workers=args.num_workers)

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    # --------- Training ----------

    # Model.
    model = TwoStreamFusion(
        args.rgb_frames, args.opf_frames, 225, 400, num_classes,
        dropout=args.dropout, freeze_base=args.freeze_base, backbone_name=args.backbone)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model = nn.DataParallel(model)

    # Loss function.
    loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(classes_loss_weights).cuda(), label_smoothing=0.1)

    # Optimizer.
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['opt_state_dict'])

    # LR scheduler. Decay LR by a factor of 0.1 every 10 epochs.
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if checkpoint is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    scaler = torch.cuda.amp.GradScaler()
    if checkpoint is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # First epoch index.
    start_epoch = 0 if checkpoint is None else checkpoint['epoch'] + 1

    train(start_epoch, dataloaders, model, loss_fn,
          optimizer, scheduler, scaler, writer, args)
