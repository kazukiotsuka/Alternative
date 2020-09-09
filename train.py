#!/usr/local/bin/python
# -*- coding:utf-8 -*-
#
# core/tts/Alternative/train.py
#

import time

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mlutils.utils import time_since, update_lr
from models.mlmodeldic import record_model, is_best_model

def train_char2wav(
        encoder, decoder, loader, model_name='newtts', lr=1e-3, n_epoch=1, seqlen=2000, stride=200,
        device=None, check_interval=1000, verbose=False, check_inference=True):
    """Train Mel2Wav Vocoder.

    params:
        model (Vocoder): instance of Vocoder
        loader (DataLoader): array of ndarray or DataLoader instance
        model_name (str): model name to be saved
        lr (float): initial learnin rate (default. 1e-4)
        n_epoch (int): num train epoch (default. 1)
        seqlen (int): sequence length to calculate loss for a training
        stride (int): stride interval to slide the learning location in a training sample
        shrink_rate (float): rate for updating seqlen and stride in every epoch
        batch_size (int): minibatch size (only when dataset is list. optional)
        device (torch.device): cuda device (optional)
        verbose (bool): if True print a lot

    returns:
        losses (list): losses
        loss_aves (list): loss averages
        model (WaveRNN) : trained model
    """
    device = torch.device(device)

    encode_optimizer = torch.optim.Adam(encoder.parameters())
    decode_optimizer = torch.optim.Adam(decoder.parameters())

    # learning rate
    for p in encode_optimizer.param_groups:
        p['lr'] = lr
    for p in decode_optimizer.param_groups:
        p['lr'] = lr

    # trim width
    init_seqlen = seqlen
    init_stride = stride

    start = time.time()
    total_iter, total_step = 0, 0
    n_data = len(loader.dataset)
    batch_size = loader.batch_size

    # r(shirink rate) gives finally 0.1 times of the seq_len and stride
    # until the last epoch.
    # 1/10 = r**epoch <=> log(r) = -1/epoch*log(10)
    shrink_rate = 10**(-1/n_epoch)

    loss_aves = []
    # epoch
    for epoch in range(n_epoch):

        # train all wavs
        for i, (wavs, targets, labels) in tqdm(enumerate(loader)):

            if verbose:
                print(f'wavs: {wavs} {wavs.shape}')
                print(f'targets: {targets} {targets.shape}')
                print(f'labels: {labels} {labels.shape}')

            # use whole sequence is too long to preserve computation graph.
            step = 0
            offset = 0
            seq_loss = 0
            T = wavs.shape[1]
            losses = []
            while offset <= T:

                # extract a part in the sequence
                if offset+seqlen < T:
                    # the sequence length is enough for seq_len
                    x = wavs[:, offset:offset+seqlen]
                    y = targets[:, offset:offset+seqlen]
                    label = labels
                else:
                    # the sequence length is not enough. all of the rest
                    x = wavs[:, offset:]
                    y = targets[:, offset:]
                    label = labels

                x = x.to(device)
                y = y.to(device)
                label = label.to(device)

                if verbose:
                    print(f'start: {offset} end:{offset+seqlen}')
                    print(f'x: {x} {x.shape}')
                    print(f'y: {y} {y.shape}')
                    print(f'label: {label} {label.shape}')

                decode_optimizer.zero_grad()
                encode_optimizer.zero_grad()

                encoder_hidden, encoder_outs = encoder(label)

                loss = 0
                for t in range(x.shape[1]):

                    if t == 0:
                        dist, h2t, h3t, phi_t, wt, kappa_t = decoder(x[:, t], label, encoder_hidden)
                    else:
                        dist, h2t, h3t, phi_t, wt, kappa_t = decoder(x[:, t], label, encoder_hidden, wt, h2t, h3t, kappa_t)

                    if verbose:
                        print(f'dist {dist} {dist.shape}')

                    loss += decoder.calculate_loss(dist, y[:, t])

                    if check_inference and t % 100 == 0:
                        print(f'sample {torch.argmax(dist, dim=1).cpu().numpy()}')
                        print(f'y {y.cpu().numpy()}')

                if verbose:
                    print(f'loss {loss.item()/ x.shape[1]}')

                # back propergation
                loss.backward()
                encode_optimizer.step()
                decode_optimizer.step()

                total_iter += 1

                # calc loss for the iteration
                #print(f'loss {loss.item()}')
                seq_loss = round(float(loss.item()), 3) / x.shape[1]

                # append to loss record
                losses += [seq_loss]
                loss_ave = np.average(losses)
                loss_aves += [loss_ave]

                # update position per step
                step += 1
                offset += stride

                if total_iter % 1 == 0:
                    print(f'epoch {epoch}/{n_epoch-1} iter: {i*batch_size}/{n_data} total_iter: {total_iter}'
                        f'-- loss ave: {loss_ave:.4f} loss: {seq_loss:.2f} '
                        f'-- elapse: {time_since(start)} speed {total_iter / (time.time() - start):.1f} steps/sec')

                if total_iter % check_interval == 0:
                    # record model
                    modeldic = record_model(
                        decoder, key_name=model_name, loss_aves=loss_aves, loss_ave=loss_ave, n_iter=total_iter,
                        settings={
                            'lr': lr, 'n_epoch': n_epoch, 'seqlen': init_seqlen, 'stride': init_stride,
                            **model.settings()},
                        model_path=(
                        f'/diskB/6/out/models/newtts/{model_name}_{model.hidden_size}_bit{model.bit}_epoch{n_epoch}_lr{lr}'
                        f'_loss{str(round(loss_ave, 3)).replace(".", "-")}'))
                    # save model if the loss average is the best score.
                    if is_best_model(
                            modeldic, key_name=model_name, compared_key='loss_ave', is_lower_better=True):
                        print(f'best score. save model.')
                        encoder.save_model(modeldic.save_model_path+'_encoder')
                        decoder.save_model(modeldic.save_model_path+'_decoder')

        # update trim width
        seqlen = int(seqlen * shrink_rate)
        stride = int(stride * shrink_rate)

        # annealing
        #update_lr(epoch, optimizer, annealing_rate=0.98, interval=1)


    return losses, loss_aves, model


def train_char2mel(
        model, loader, model_name='char2mel', lr=1e-3, n_epoch=1,
        device=None, check_interval=1000, verbose=False):
    """Train Char2Mel.

    params:
        model (Char2Mel): instance of Char2Mel
        loader (DataLoader): array of ndarray or DataLoader instance
        model_name (str): model name to be saved
        lr (float): initial learnin rate (default. 1e-4)
        n_epoch (int): num train epoch (default. 1)
        batch_size (int): minibatch size (only when dataset is list. optional)
        device (torch.device): cuda device (optional)
        verbose (bool): if True print a lot

    returns:
        losses (list): losses
        loss_aves (list): loss averages
        model (WaveRNN) : trained model
    """
    device = torch.device(device)

    optimizer = torch.optim.Adam(model.parameters())

    # learning rate
    for p in optimizer.param_groups:
        p['lr'] = lr

    start = time.time()
    losses, loss_aves = [], []
    total_iter, total_step = 0, 0
    n_data = len(loader.dataset)
    batch_size = loader.batch_size

    # epoch
    for epoch in range(n_epoch):
        # train all wavs
        for i, (mels, labels) in enumerate(loader):

            if verbose:
                print(f'mels: {mels} {mels.shape}')
                print(f'labels: {labels} {labels.shape}')

            optimizer.zero_grad()

            mels = mels.to(device)
            labels = labels.to(device)

            input_features = mels[:, 1:]
            target_features = mels[:, :-1]

            predict = model(input_features, labels)

            # print(f'predict {predict} {predict.shape}')
            # print(f'target {target_features} {target_features.shape}')

            loss = model.calculate_loss(predict, target_features)

            # back propergation
            loss.backward()
            optimizer.step()

            total_iter += 1

            # append to loss record
            seq_loss = float(loss.item())
            losses += [seq_loss]
            loss_ave = np.average(losses)
            loss_aves += [loss_ave]

            if total_iter % 100 == 0:
                print(f'epoch {epoch}/{n_epoch-1} iter: {i*batch_size}/{n_data} total_iter: {total_iter}'
                      f'-- loss ave: {loss_ave:.4f} loss: {seq_loss:.2f} '
                      f'-- elapse: {time_since(start)} speed {total_iter/(time.time() - start):.1f} steps/sec')

            if total_iter % check_interval == 0:
                # record model
                modeldic = record_model(
                    model, key_name=model_name, loss_aves=loss_aves, loss_ave=loss_ave, n_iter=len(losses),
                    settings={'lr': lr, 'n_epoch': n_epoch, **model.settings()},
                    model_path=(
                    f'/diskB/6/out/models/char2mel/{model_name}_epoch{n_epoch}_lr{lr}'
                    f'_loss{str(round(loss_ave, 3)).replace(".", "-")}'))
                # save model if the loss average is the best score.
                if is_best_model(
                        modeldic, key_name=model_name, compared_key='loss_ave', is_lower_better=True):
                    print(f'best score. save model.')
                    model.save_model(modeldic.save_model_path)

                # annealing
                #update_lr(i, optimizer, annealing_rate=0.98, interval=5000)


    return losses, loss_aves, model


def train_vocoder(
        model, loader, model_name='vocoder', lr=1e-3, n_epoch=1, seqlen=5000, stride=500,
        device=None, check_interval=1000, verbose=False, check_inference=False):
    """Train Mel2Wav Vocoder.

    params:
        model (Vocoder): instance of Vocoder
        loader (DataLoader): array of ndarray or DataLoader instance
        model_name (str): model name to be saved
        lr (float): initial learnin rate (default. 1e-4)
        n_epoch (int): num train epoch (default. 1)
        seqlen (int): sequence length to calculate loss for a training
        stride (int): stride interval to slide the learning location in a training sample
        shrink_rate (float): rate for updating seqlen and stride in every epoch
        batch_size (int): minibatch size (only when dataset is list. optional)
        device (torch.device): cuda device (optional)
        verbose (bool): if True print a lot

    returns:
        losses (list): losses
        loss_aves (list): loss averages
        model (WaveRNN) : trained model
    """
    device = torch.device(device)

    optimizer = torch.optim.Adam(model.parameters())

    # learning rate
    for p in optimizer.param_groups:
        p['lr'] = lr

    # trim width
    init_seqlen = seqlen
    init_stride = stride

    start = time.time()
    total_iter, total_step = 0, 0
    n_data = len(loader.dataset)
    batch_size = loader.batch_size

    # r(shirink rate) gives finally 0.1 times of the seq_len and stride
    # until the last epoch.
    # 1/10 = r**epoch <=> log(r) = -1/epoch*log(10)
    shrink_rate = 10**(-1/n_epoch)

    loss_aves = []
    # epoch
    for epoch in range(n_epoch):

        # train all wavs
        for i, (wavs, targets, mels) in enumerate(loader):

            if verbose:
                print(f'wavs: {wavs} {wavs.shape}')
                print(f'targets: {targets} {targets.shape}')
                print(f'mels: {mels} {mels.shape}')

            # use whole sequence is too long to preserve computation graph.
            step = 0
            offset = 0
            T = wavs.shape[1]
            losses = []
            while offset <= T:
                optimizer.zero_grad()

                # extract a part in the sequence
                if offset+seqlen < T:
                    # the sequence length is enough for seq_len
                    x = wavs[:, offset:offset+seqlen]
                    y = targets[:, offset:offset+seqlen]
                    mel = mels[:, offset:offset+seqlen, :]
                else:
                    # the sequence length is not enough. all of the rest
                    x = wavs[:, offset:]
                    y = targets[:, offset:]
                    mel = mels[:, offset:, :]

                x = x.to(device)
                y = y.to(device)
                mel = mel.to(device)

                if verbose:
                    print(f'start: {offset} end:{offset+seqlen}')
                    print(f'x: {x} {x.shape}')
                    print(f'y: {y} {y.shape}')
                    print(f'mel: {mel} {mel.shape}')

                predict = model(x, mel)

                if verbose:
                    print(f'predict {predict} {predict.shape}')

                # cross entropy loss
                loss = model.calculate_loss(predict, y)

                # back propergation
                loss.backward()
                optimizer.step()

                # back propergation loss.backward() optimizer.step()
                total_iter += 1

                # update position per step
                step += 1
                offset += stride

                # append to loss record
                seq_loss = float(loss.item())
                losses += [seq_loss]

                loss_ave = np.average(losses)

                if check_inference and total_iter % 500 == 0:
                    print(f'samples {torch.argmax(predict, dim=2).cpu().numpy()}')
                    print(f'y {y.cpu().numpy()}')
                    print(f'loss {loss.item()}')

                if total_iter % 100 == 0:
                    print(f'epoch {epoch}/{n_epoch-1} iter: {i*batch_size}/{n_data} total_iter: {total_iter}'
                          f'-- loss ave: {loss_ave:.4f} loss: {seq_loss:.2f} '
                          f'-- elapse: {time_since(start)} speed {total_iter / (time.time() - start):.1f} steps/sec')

                if total_iter % check_interval == 0 and total_iter != 0:
                    # record model
                    modeldic = record_model(
                        model, key_name=model_name, loss_aves=loss_aves, loss_ave=loss_ave, n_iter=total_iter,
                        settings={
                            'lr': lr, 'n_epoch': n_epoch, 'seqlen': init_seqlen, 'stride': init_stride,
                            **model.settings()},
                        model_path=(
                        f'/diskB/6/out/models/vocoder/{model_name}_{model.hidden_size}_bit{model.bit}_epoch{n_epoch}_lr{lr}'
                        f'_loss{str(round(loss_ave, 3)).replace(".", "-")}'))
                    # save model if the loss average is the best score.
                    if is_best_model(
                            modeldic, key_name=model_name, compared_key='loss_ave', is_lower_better=True):
                        print(f'best score. save model.')
                        model.save_model(modeldic.save_model_path)
                    if total_iter % check_interval * 10 == 0:
                        print(f'worked hard. save model.')
                        model.save_model(modeldic.save_model_path)

            loss_aves += [loss_ave]

        # update trim width
        seqlen = int(seqlen * shrink_rate)
        stride = int(stride * shrink_rate)

        # annealing
        #update_lr(epoch, optimizer, annealing_rate=0.98, interval=1)


    return losses, loss_aves, model