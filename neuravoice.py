#!/usr/local/bin/python
# -*- coding:utf-8 -*-

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.phoneme import Phoneme43
from tts.duration.phoneme_duration import PhonemeDuration
from tools.libaudio.encodes import mulaw_decode
from mlutils.nn import NGramConvolution
from mlutils.utils import to_onehot


class AttentionKGaussianWindow(nn.Module):
    def __init__(self, K=10, hidden_size=256, device='cuda:0', version=0):
        super(AttentionKGaussianWindow, self).__init__()
        self.__device__ = device
        self.Wh1p = nn.Linear(hidden_size, 3 * K)
        self.K = K
        self.epsilon = 1e-5
        self.version = version
        #self.bp = nn.Parameter(torch.zeros(3 * K)) Linear includes bias

        # send weights to device
        self.to_device(device)

    def to_device(self, device=None):
        assert device and (device.startswith('cuda') or device.startswith('cpu')), f'invalid device {device}'
        device = torch.device(device)
        self.Wh1p.to(device)

    def forward(self, h1t_1, kappa_t_1, cx):
        """Compute Attention Window.

        h1t: (batch, hidden_size)  Wh1p: (hidden_size, 3K) -> out: (batch, 3K)
        -> alpha, beta, k : (batch, K) x 3

        args:
            - h1t of shape (batch, hidden_size): hidden state from the first hidden layer
            - kt_1: previous kappa
            - cx (N, T, H):

        returns:
            - wt: window of time t
            - kt: kappa of time t
            - phi: phi of time t
        """
        #import pdb; pdb.set_trace()

        # (αhat_t, βhat_t, khat_t) = Wh1p ht^1 + bp  # output of the first hidden layer
        alpha_beta_kappa = self.Wh1p(h1t_1)# + self.bp
        K = self.K
        alpha_t, beta_t, kappa_t = alpha_beta_kappa[:, :K], \
                                   alpha_beta_kappa[:, K:K * 2], \
                                   alpha_beta_kappa[:, K*2:K*3]

        # αt = exp(αhat_t)  # importance of the window
        # βt = exp(βhat_t)  # width of the window
        # κt = κt-1 + exp(κhat_t)  # location of the window (how far to slide each window)
        alpha_t = alpha_t.exp() + self.epsilon
        beta_t = beta_t.exp() + self.epsilon
        kappa_t = kappa_t_1 + kappa_t.exp()

        # φ(t, u) = Σk=1->K αkt*exp(-βkt(κkt-u)^2)  # mixture of K Gaussian
        u = torch.Tensor(range(cx.shape[1])).to(torch.device(self.__device__))
        phi_t = (alpha_t.unsqueeze(2) *
                 (-beta_t.unsqueeze(2) * (kappa_t.unsqueeze(2).repeat(1, 1, cx.shape[1]) - u) ** 2).exp()).sum(dim=1)

        # The size of the soft window vectors is the same as the size of the character vectors
        # cu (assuming a one-hot encoding, this will be the number of characters in the alphabet).
        wt = (cx.float() * phi_t.unsqueeze(2)).sum(dim=1)

        return wt, kappa_t, phi_t


class InputEncoder(nn.Module):
    def __init__(
            self, encode_type: ('onehot', 'embed', 'ngc', 'rnn') = 'onehot',
            embed_size=256, hidden_size=256):
        super(InputEncoder, self).__init__()
        self.vocab_size = len(Phoneme43)
        self.encode_type = encode_type

        # embedding
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        # n-gram conv (ngc)
        self.ngc = NGramConvolution(embed_size=embed_size, kernel_size=(5, 1))
        # rnn
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, x):
        """
        args:

        """
        if len(x.shape) == 1:
            x = torch.LongTensor(to_onehot(phonemes, n_class=vocab_size))

        if self.encode_type in ('embed', 'ngc', 'rnn'):
            x = self.embedding(x)
        if self.encode_type in ('ngc', 'rnn'):
            x = ngc(x_emb)
        if self.encode_type == 'rnn':
            out, h = self.gru()

        if self.encode_type == 'rnn':
            return out, h
        else:
            return x


class CharToMel(nn.Module):
    def __init__(
            self, encode_type: ('onehot', 'embed', 'ngc', 'rnn') = 'onehot',
            version=1,
            K=20, hidden_size=512, feature_size=128, out_size=512, device='cuda:0'):
        super(CharToMel, self).__init__()
        self.__device__ = device
        self.encode_type = encode_type
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.out_size = out_size
        self.K = K
        if version < 6:
            self.encoder = InputEncoder(encode_type=encode_type, hidden_size=hidden_size)
            self.vocab_size = self.window_size = self.encoder.vocab_size
        else:
            self.vocab_size = self.window_size = len(Phoneme43)
        self.H1 = nn.GRUCell(input_size=self.feature_size + self.window_size, hidden_size=hidden_size)
        self.H2 = nn.GRUCell(input_size=self.feature_size + self.window_size + hidden_size,
                                   hidden_size=hidden_size)
        self.H3 = nn.GRUCell(input_size=self.feature_size + self.window_size + hidden_size,
                                   hidden_size=hidden_size)
        self.window = AttentionKGaussianWindow(K=K, hidden_size=hidden_size, device=device, version=version)
        self.Wh1y = nn.Linear(hidden_size, out_size)
        self.Wh2y = nn.Linear(hidden_size, out_size)
        self.Wh3y = nn.Linear(hidden_size, out_size)
        self.Y = nn.Linear(self.out_size, feature_size)
        self.mse_loss = nn.MSELoss()

        self.version = version
        # optional units
        if version == 1 or version == 2:
            self.train_init_params = False
        if version == 3:
            self.train_init_params = True
            self.nonlinear = ''
        if version == 4:
            self.train_init_params = True
            self.nonlinear = 'relu'
        if version == 5:
            self.train_init_params = True
            self.nonlinear = ''
        if version == 6:
            self.train_init_params = True
            self.nonlinear = ''
        if version == 7:
            # modify window calc
            # use gru not cell
            self.train_init_params = True
            self.nonlinear = ''
        if version >= 8:
            self.train_init_params = True

        # trainable initial parameters
        if self.train_init_params:
            self.h1_0 = nn.Parameter(torch.zeros((hidden_size)))
            self.h2_0 = nn.Parameter(torch.zeros((hidden_size)))
            self.h3_0 = nn.Parameter(torch.zeros((hidden_size)))
            self.w_0 = nn.Parameter(torch.zeros((self.vocab_size)))

        # send weights to device
        self.to_device(device)

        # print summary
        self.print_summary()
    def to_device(self, device=None):
        #if device.startswith('cpu'):
        #    self.__device__ = device
        #elif not device.startswith('cuda'):
        assert device and (device.startswith('cuda') or device.startswith('cpu')), f'invalid device {device}'
        device = torch.device(device)
        if self.version < 5:
            self.encoder.to(device)
        self.window.to(device)
        self.H1.to(device), self.H2.to(device), self.H3.to(device)
        #if self.version >= 7:
        #    self.rnn1.to(device), self.rnn2.to(device), self.rnn3.to(device)
        self.Wh1y.to(device), self.Wh2y.to(device), self.Wh3y.to(device)
        self.Y.to(device),
        self.mse_loss.to(device)
        if self.train_init_params:
            self.h1_0.to(device), self.h2_0.to(device), self.h3_0.to(device), self.w_0.to(device)
        # self.w_to_out.to(device), self.h_to_y.to(device)

    def initial_states(self, batch_size):
        h1_0 = torch.zeros((batch_size, self.hidden_size)).to(self.__device__)
        h2_0 = torch.zeros((batch_size, self.hidden_size)).to(self.__device__)
        h3_0 = torch.zeros((batch_size, self.hidden_size)).to(self.__device__)
        kappa_0 = torch.zeros((batch_size, self.K)).to(self.__device__)
        w_0 = torch.zeros((batch_size, self.vocab_size)).to(self.__device__)  # TODO: window size
        #if not self.__device__.startswith('cpu'):
        #    h1_0, h2_0, h3_0, kappa_0, w_0 = \
        #        h1_0.cuda(self.__device__), h2_0.cuda(self.__device__), \
        #        h3_0.cuda(self.__device__), kappa_0.cuda(self.__device__), \
        #        w_0.cuda(self.__device__)
        return h1_0, h2_0, h3_0, kappa_0, w_0

    def forward(self, x, c, verbose=False):
        """
        args:
            - x (N, T, H) : mel features
            - c (N, U, V) : character sequence (U:character sequence length), V:vocabulary size)
        """

        # TODO: test encode pattens
        # if self.encode_type in ('onehot') and c.dim() < 3:
        #    encode_out = self.encoder(c)

        #import pdb; pdb.set_trace()

        N = x.shape[0]
        T = x.shape[1]
        H = x.shape[2]
        U = c.shape[1]
        V = c.shape[2]

        if verbose: print(f'batch size: {N} time step: {T} feature size: {H} character sequence step: {U} vocabulary size: {V}')

        # set previous states by initial states
        if self.train_init_params:
            h1t_1, h2t_1, h3t_1, wt_1 = \
                self.h1_0.repeat(1, 1, N).view(N, -1).to(self.__device__),\
                self.h2_0.repeat(1, 1, N).view(N, -1).to(self.__device__),\
                self.h3_0.repeat(1, 1, N).view(N, -1).to(self.__device__),\
                self.w_0.repeat(1, 1, N).view(N, -1).to(self.__device__)
            kappa_t_1 = torch.zeros((N, self.K)).to(self.__device__)
        else:
            h1t_1, h2t_1, h3t_1, kappa_t_1, wt_1 = self.initial_states(N)

        h1, h2, h3, w = [], [], [], []
        for t in range(T):
            h1t = self.H1(torch.cat([x[:, t, :], wt_1], dim=1), h1t_1)
            wt, kappa_t, phi_t = self.window.forward(h1t, kappa_t_1, c)
            h2t = self.H2(torch.cat([x[:, t, :], wt, h1t], dim=1), h2t_1)
            h3t = self.H3(torch.cat([x[:, t, :], wt, h2t], dim=1), h3t_1)
            h1t_1, h2t_1, h3t_1, wt_1, kappa_t_1 = h1t, h2t, h3t, wt, kappa_t
            h1 += [h1t]  # (N, T, hidden_size)
            h2 += [h2t]
            h3 += [h3t]
            w += [wt]

        h1, h2, h3, w = \
            torch.stack(h1, dim=1), torch.stack(h2, dim=1),\
            torch.stack(h3, dim=1), torch.stack(w, dim=1)

        # by + Σn=1->N Whny hnt
        # MEMO: or use only the last state h3
        h = self.Wh1y(h1) + self.Wh2y(h2) + self.Wh3y(h3)# + self.w_to_out(w)
        # yt = Y(yˆt)
        y = self.Y(h)
        return y

    def generate(self, phonemes:list, durations:list = None, duration_model:PhonemeDuration = None, sample_rate=24000):
        """Generate Sequence.

        args:
            - phonemes (list): list of Phoneme43 index
            - durations (list, optional): result of duration model
            - duration_model (DurationModel, optional):
        """
        #import pdb; pdb.set_trace()

        with torch.no_grad():
            # onehot encoding
            c = torch.LongTensor(
                    to_onehot(phonemes, n_class=self.vocab_size)).unsqueeze(0).to(self.__device__)

            # batch size is always 1
            N = 1
            if durations is not None:
                T = int((sum(durations)/1000)*sample_rate // 299.19)
            elif duration_model:
                T = int((sum(duration_model.predict(phonemes)[1])/1000)*sample_rate // 299.19)
            else:
                assert False, 'either duration_model or durations are necessary.'

            # initial states
            h1t_1, h2t_1, h3t_1, kappa_t_1, wt_1 = self.initial_states(N)
            x = torch.zeros((N, self.feature_size)).to(self.__device__)

            features = []
            phi = []
            w = []
            for t in range(T):
                h1t = self.H1(torch.cat([x, wt_1], dim=1), h1t_1)
                wt, kappa_t, phi_t = self.window.forward(h1t, kappa_t_1, c)
                h2t = self.H2(torch.cat([x, wt, h1t], dim=1), h2t_1)
                h3t = self.H3(torch.cat([x, wt, h2t], dim=1), h3t_1)
                h = self.Wh1y(h1t) + self.Wh2y(h2t) + self.Wh3y(h3t) # + self.w_to_out(wt)
                y = self.Y(h)

                # update previous states
                h1t_1, h2t_1, h3t_1, wt_1, kappa_t_1 = h1t, h2t, h3t, wt, kappa_t
                x = y

                features += [y]
                phi += [phi_t]
                w += [wt]

            if not self.__device__.startswith('cpu'):
                return torch.stack(features, dim=0).squeeze().cpu().detach().numpy(), \
                       torch.stack(phi, dim=0).squeeze().cpu().detach().numpy(), \
                       torch.stack(w, dim=0).squeeze().cpu().detach().numpy()
            else:
                return torch.stack(features, dim=0).squeeze().detach().numpy(),\
                       torch.stack(phi, dim=0).squeeze().detach().numpy(), \
                       torch.stack(w, dim=0).squeeze().detach().numpy()

    def calculate_loss(self, predict, target):
        """Calculate Loss.

        MSE
            ℓ(x,y)=L={l1,...lN}T, ln = (xn-yn)2

        args:
            predict (N, T, H) : predicted features
            target (N, T, H) : target features
        """
        #import pdb; pdb.set_trace()
        return self.mse_loss(predict, target)

    def print_summary(self) :
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print(f'model version {self.version}')
        print('Trainable Parameters: %.3f million' % parameters)

    def settings(self) -> dict:
        return {
            'encode_type': self.encode_type,
            'hidden_size': self.hidden_size,
            'K': self.K,
            'out_size': self.out_size
        }

    @classmethod
    def init_from_settings(cls, settings, model_path, device, **kwargs):
        assert device and (device.startswith('cuda') or device.startswith('cpu')), f'invalid device {device}'
        setting_params = ['encode_type', 'K', 'hidden_size', 'out_size']
        settings = {k:v for k, v in settings.items() if k in setting_params}
        assert all([key in settings.keys() for key in setting_params])
        model = cls(**{**settings,**kwargs,**{'device': device}})
        return model.load_model(model_path)

    def save_model(self, save_model_path: str):
        try:
            print(f'trying to save model parameters {self.state_dict().keys()} to {save_model_path} ..')
            torch.save(self.state_dict(), save_model_path)
            # torch.save(self, save_model_path)  # * this fails when data parallel
        except Exception as e:
            print(e)

    def load_model(self, model_file_path: str):
        try:
            self.load_state_dict(
                torch.load(model_file_path, map_location=lambda storage, loc: storage))
            # torch.load(model_file_path)  # * this fails if trained on multiple GPU. use state dict.
            return self
        except Exception as e:
            print(e)



class Vocoder(nn.Module):
    """Vocoder generates wav using mel spectrogram.

    WaveRNN math::
        xt = [ct-1, ft-1, ct]  # input
        ut = σ(Ru ht-1 + Iu*xt + bu)  # update gate
        rt = σ(Rr ht-1 + Ir*xt + br)  # reset gate
        et = tanh(rt∘(Re ht-1) + Ie*xt + be)  # recurrent unit
        ht = ut∘ht-1 + (1-u)∘et  # next hidden state
        yc, yf = split(ht)  # coarse, fine
        P(ct) = softmax(O2 relu(O1 yc))  # coarse distribution
        P(ft) = softmax(O4 relu(O3 yf))  # fine distribution
    """

    def __init__(self,
            feature_size=128, ax_size=0, hidden_size=512, bit=9, out_size=512,
            version=2,
            sample_rate=24000, device='cuda:0'):
        super(Vocoder, self).__init__()

        self.__device__ = device
        self.__disable_cuda__ = device.startswith('cpu')

        self.version = version
        self.hidden_size = hidden_size
        self.bit = bit
        self.n_class = 2**bit
        self.sample_rate = sample_rate
        self.feature_size = feature_size  # mel feature size
        self.ax_size = 0
        #self.gru = nn.GRU(input_size=3+feature_size+ax_size, hidden_size=hidden_size)
        #self.O1 = nn.Linear(hidden_size//2, hidden_size//2)
        #self.O2 = nn.Linear(hidden_size//2, hidden_size//2)
        #self.O3 = nn.Linear(hidden_size//2, self.n_class)
        #self.O4 = nn.Linear(hidden_size//2, self.n_class)

        self.I = nn.Linear(feature_size+1, hidden_size)
        self.H1 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.H2 = nn.GRU(input_size=hidden_size+ax_size, hidden_size=hidden_size)

        self.O1 = nn.Linear(hidden_size, hidden_size)
        self.O2 = nn.Linear(hidden_size, self.n_class)

        # logsoftmax
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # criterion
        self.criterion = nn.CrossEntropyLoss()

        # print summary
        self.print_summary()

        # to specific device
        if device: self.to_device(device)

    def to_device(self, device=None):
        assert device and (device.startswith('cuda') or device.startswith('cpu')), f'invalid device {device}'
        device = torch.device(device)
        self.I.to(device), self.H1.to(device), self.H2.to(device), self.O1.to(device), self.O2.to(device)
        self.criterion.to(device)

    def forward(self, x, mel, ax=None):
        """Forward step.

        args:
            - x (torch.FloatTensor): wav (N, T)
            - mel (torch.FloatTensor): mel feature (N, T, feature_size)
            - ax (torch.FloatTensor): auxilialy
        """
        #import pdb; pdb.set_trace()

        N = x.shape[0]
        T = x.shape[1]
        h1 = torch.zeros(1, T, self.hidden_size).to(self.__device__)
        h2 = torch.zeros(1, T, self.hidden_size).to(self.__device__)

        x = torch.cat([x.unsqueeze(-1), mel], dim=2)

        x = self.I(x)
        res = x

        x, _ = self.H1(x, h1)
        x = x + res
        res = x

        if ax:
            x = torch.cat([x, ax], dim=2)
        x, _ = self.H2(x, h2)
        x = x + res

        x = torch.relu(self.O1(x))
        x = torch.relu(self.O2(x))

        return x

    def calculate_loss(self, x, y):
        #import pdb; pdb.set_trace()
        return self.criterion(x.transpose(1, 2), y.squeeze(-1).long())

    def prepare_generation(self):
        # gru 1
        self.h1 = nn.GRUCell(self.H1.input_size, self.H1.hidden_size)
        self.h1.weight_hh.data = self.H1.weight_hh_l0.data
        self.h1.weight_ih.data = self.H1.weight_ih_l0.data
        self.h1.bias_hh.data = self.H1.bias_hh_l0.data
        self.h1.bias_ih.data = self.H1.bias_ih_l0.data
        # gru 2
        self.h2 = nn.GRUCell(self.H2.input_size, self.H2.hidden_size)
        self.h2.weight_hh.data = self.H2.weight_hh_l0.data
        self.h2.weight_ih.data = self.H2.weight_ih_l0.data
        self.h2.bias_hh.data = self.H2.bias_hh_l0.data
        self.h2.bias_ih.data = self.H2.bias_ih_l0.data

    def generate(self, mel, ax=None, parallel=1, return_descriptions=False):
        """Generate a signal.

        args:
            - mel (torch.FloatTensor): (N, H, T)
        """
        #import pdb; pdb.set_trace()

        self.eval()
        if not hasattr(self, 'H'):
            self.prepare_generation()

        T = int(mel.shape[2] * 299.19)
        fold_len = T//parallel
        overlap = 100

        start = time.time()
        speed = None

        with torch.no_grad():
            # upsample mel
            mel = torch.nn.functional.interpolate(
                mel, T, mode='linear', align_corners=True).transpose(1,2).to(self.__device__)

            if parallel > 1:
                mel = self.fold_with_overlap(mel, fold_len, overlap)

            N = mel.shape[0]

            # starting samples
            x = torch.zeros((N, 1)).to(self.__device__)  # + 128.
            h1 = torch.zeros(N,  self.hidden_size).to(self.__device__)
            h2 = torch.zeros(N,  self.hidden_size).to(self.__device__)

            # generated samples
            samples = []

            for t in range(fold_len):

                x = torch.cat([x, mel[:, t, :]], dim=1)

                x = self.I(x)
                res = x

                h1 = self.h1(x, h1)
                x = h1 + res
                res = x

                if ax:
                    x = torch.cat([x, ax], dim=1)
                h2 = self.h2(x, h2)
                x = h2 + res

                x = torch.relu(self.O1(x))
                x = torch.relu(self.O2(x))

                x = torch.argmax(x, dim=1, keepdim=True).float()

                samples += [x]

                # time check
                if t % 1000 == 0 and t != 0:
                    speed = N * (t + 1) / (time.time() - start)
                    print(f'generate {(t+1)*N}/{T}, batch {N}, Speed: {speed:.2f} samples/sec, '
                          f'x_realtime: {round(speed/self.sample_rate, 3)}')

        #import pdb; pdb.set_trace()

        # convert to array and decode
        samples = torch.stack(samples, dim=0)
        if parallel > 1:
            samples = mulaw_decode(samples.transpose(0,1).squeeze().cpu().numpy())
            sample = self.xfade_and_unfold(samples, fold_len, overlap)
        else:
            sample = mulaw_decode(samples.squeeze().cpu().numpy())

        if return_descriptions:
            samples_per_sec = round(speed, 3)
            batch = N
            total_samples = fold_len * batch
            x_realtime = round(speed/self.sample_rate, 3)
            return sample, samples_per_sec, batch, total_samples, x_realtime
        else:
            return sample


    def print_summary(self) :
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)

    def settings(self) -> dict:
        return {
            'hidden_size': self.hidden_size,
            'bit': self.bit,
            'sample_rate': self.sample_rate,
            'ax_size': self.ax_size,
        }

    @classmethod
    def init_from_settings(cls, settings, model_path, device, **kwargs):
        assert device and (device.startswith('cuda') or device.startswith('cpu')), f'invalid device {device}'
        setting_params = ['hidden_size', 'sample_rate', 'bit', 'ax_size']
        settings = {k:v for k, v in settings.items() if k in setting_params}
        assert all([key in settings.keys() for key in setting_params])
        model = cls(**{**settings,**kwargs,**{'device': device}})
        model.load_model(model_path)
        return model

    def save_model(self, save_model_path: str):
        try:
            print(f'trying to save model parameters {self.state_dict().keys()} to {save_model_path} ..')
            torch.save(self.state_dict(), save_model_path)
            # torch.save(self, save_model_path)  # * this fails when data parallel
        except Exception as e:
            print(e)

    def load_model(self, model_file_path: str):
        try:
            self.load_state_dict(
                torch.load(model_file_path, map_location=lambda storage, loc: storage))
            # torch.load(model_file_path)  # * this fails if trained on multiple GPU. use state dict.
        except Exception as e:
            print(e)

    # https://github.com/fatchord/WaveRNN/
    def fold_with_overlap(self, x, target, overlap):

        ''' Fold the tensor with overlap for quick batched inference.
            Overlap will be used for crossfading in xfade_and_unfold()

        Args:
            x (tensor)    : Upsampled conditioning features.
                            shape=(1, timesteps, features)
            target (int)  : Target timesteps for each index of batch
            overlap (int) : Timesteps for both xfade and rnn warmup

        Return:
            (tensor) : shape=(num_folds, target + 2 * overlap, features)

        Details:
            x = [[h1, h2, ... hn]]

            Where each h is a vector of conditioning features

            Eg: target=2, overlap=1 with x.size(1)=10

            folded = [[h1, h2, h3, h4],
                      [h4, h5, h6, h7],
                      [h7, h8, h9, h10]]
        '''

        _, total_len, features = x.size()

        # Calculate variables needed
        num_folds = (total_len - overlap) // (target + overlap)
        extended_len = num_folds * (overlap + target) + overlap
        remaining = total_len - extended_len

        # Pad if some time steps poking out
        if remaining != 0:
            num_folds += 1
            padding = target + 2 * overlap - remaining
            x = self.pad_tensor(x, padding, side='after')

        folded = torch.zeros(num_folds, target + 2 * overlap, features).to(self.__device__)

        # Get the values for the folded tensor
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            folded[i] = x[:, start:end, :]

        return folded

    # https://github.com/fatchord/WaveRNN/
    def xfade_and_unfold(self, y, target, overlap):

        ''' Applies a crossfade and unfolds into a 1d array.

        Args:
            y (ndarry)    : Batched sequences of audio samples
                            shape=(num_folds, target + 2 * overlap)
                            dtype=np.float64
            overlap (int) : Timesteps for both xfade and rnn warmup

        Return:
            (ndarry) : audio samples in a 1d array
                       shape=(total_len)
                       dtype=np.float64

        Details:
            y = [[seq1],
                 [seq2],
                 [seq3]]

            Apply a gain envelope at both ends of the sequences

            y = [[seq1_in, seq1_target, seq1_out],
                 [seq2_in, seq2_target, seq2_out],
                 [seq3_in, seq3_target, seq3_out]]

            Stagger and add up the groups of samples:

            [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]

        '''

        num_folds, length = y.shape
        target = length - 2 * overlap
        total_len = num_folds * (target + overlap) + overlap

        # Need some silence for the rnn warmup
        silence_len = overlap // 2
        fade_len = overlap - silence_len
        silence = np.zeros((silence_len), dtype=np.float64)

        # Equal power crossfade
        t = np.linspace(-1, 1, fade_len, dtype=np.float64)
        fade_in = np.sqrt(0.5 * (1 + t))
        fade_out = np.sqrt(0.5 * (1 - t))

        # Concat the silence to the fades
        fade_in = np.concatenate([silence, fade_in])
        fade_out = np.concatenate([fade_out, silence])

        # Apply the gain to the overlap samples
        y[:, :overlap] *= fade_in
        y[:, -overlap:] *= fade_out

        unfolded = np.zeros((total_len), dtype=np.float64)

        # Loop to add up all the samples
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            unfolded[start:end] += y[i]

        return unfolded

    # https://github.com/fatchord/WaveRNN
    def pad_tensor(self, x, pad, side='both') :
        # NB - this is just a quick method i need right now
        # i.e., it won't generalise to other shapes/dims
        b, t, c = x.size()
        total = t + 2 * pad if side == 'both' else t + pad
        padded = torch.zeros(b, total, c).to(self.__device__)
        if side == 'before' or side == 'both' :
            padded[:, pad:pad+t, :] = x
        elif side == 'after' :
            padded[:, :t, :] = x
        return padded