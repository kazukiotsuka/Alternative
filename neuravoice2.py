import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.phoneme import Phoneme43
from tts.duration.phoneme_duration import PhonemeDuration
from tools.libaudio.encodes import combine_signal
from mlutils.nn import NGramConvolution
from mlutils.utils import to_onehot

class Encoder(nn.Module):
    def __init__(
            self, version=1, sample_rate=24000,
            K=10, hidden_size=512, bit=9, device='cuda:0'):
        super(Encoder, self).__init__()
        self.__device__ = device

        #self.feature_size = feature_size
        self.K = K
        self.vocab_size = self.window_size = len(Phoneme43)
        self.bit = bit
        self.n_class = 2**bit
        self.sample_rate = sample_rate
        self.hidden_size = hidden_size
        self.version = version

        #self.H1 = nn.GRUCell(input_size=self.feature_size + self.window_size, hidden_size=hidden_size)
        #self.H2 = nn.GRUCell(input_size=self.feature_size + self.window_size + hidden_size,
        #                     hidden_size=hidden_size)
        #self.H3 = nn.GRUCell(input_size=self.feature_size + self.window_size + hidden_size,
        #                     hidden_size=hidden_size)

        self.Wh1p = nn.Linear(self.hidden_size + self.window_size, self.K*3)
        self.rnn1 = nn.GRU(input_size=self.vocab_size, hidden_size=hidden_size)

        # optional units
        self.train_init_params = True
        self.nonlinear = ''

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
        # if device.startswith('cpu'):
        #    self.__device__ = device
        # elif not device.startswith('cuda'):
        assert device and (device.startswith('cuda') or device.startswith('cpu')), f'invalid device {device}'
        device = torch.device(device)
        #self.window.to(device)
        #self.H1.to(device), self.H2.to(device), self.H3.to(device)
        self.rnn1.to(device)
        #self.Wh1y.to(device), self.Wh2y.to(device), self.Wh3y.to(device)
        self.Wh1p.to(device)
        self.rnn1.to(device)
        if self.train_init_params:
            self.h1_0.to(device), self.h2_0.to(device), self.h3_0.to(device), self.w_0.to(device)
        # self.w_to_out.to(device), self.h_to_y.to(device)

    def initial_states(self, batch_size):
        if self.train_init_params:
            h1_0 = self.h1_0.repeat(1, 1, batch_size).view(1, batch_size , -1).to(self.__device__)
        else:
            h1_0 = torch.zeros((1, batch_size, self.hidden_size)).to(self.__device__)
        return h1_0

    def forward(self, cx):
        #import pdb; pdb.set_trace()

        N = cx.shape[0]
        U = cx.shape[1]
        V = cx.shape[2]

        h1 = self.initial_states(N)

        # encoder rnn
        out1, h1 = self.rnn1(cx.float().transpose(0, 1), h1)
        encoder_outs = out1
        encoder_hidden = h1.squeeze(1)

        return encoder_hidden, encoder_outs

    def print_summary(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print(f'model version {self.version}')
        print('Trainable Parameters: %.3f million' % parameters)


    @classmethod
    def init_from_settings(cls, settings, model_path, device, **kwargs):
        assert device and (device.startswith('cuda') or device.startswith('cpu')), f'invalid device {device}'
        setting_params = ['K', 'hidden_size', 'sample_rate', 'bit']
        settings = {k: v for k, v in settings.items() if k in setting_params}
        assert all([key in settings.keys() for key in setting_params])
        model = cls(**{**settings, **kwargs, **{'device': device}})
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


class NewTTS(nn.Module):
    def __init__(
            self, version=1, sample_rate=24000,
            K=10, hidden_size=512, bit=9, device='cuda:0'):
        super(NewTTS, self).__init__()
        self.__device__ = device

        #self.feature_size = feature_size
        self.K = K
        self.vocab_size = self.window_size = len(Phoneme43)
        self.bit = bit
        self.n_class = 2**bit
        self.sample_rate = sample_rate
        self.hidden_size = hidden_size

        self.epsilon = 1e-5

        #self.encoder = Encoder(version=version, sample_rate=sample_rate, K=K, hidden_size=hidden_size, bit=bit, device=device)
        #self.H1 = nn.GRUCell(input_size=self.feature_size + self.window_size, hidden_size=hidden_size)
        #self.H2 = nn.GRUCell(input_size=self.feature_size + self.window_size + hidden_size,
        #                     hidden_size=hidden_size)
        #self.H3 = nn.GRUCell(input_size=self.feature_size + self.window_size + hidden_size,
        #                     hidden_size=hidden_size)

        self.Wh1p = nn.Linear(self.hidden_size + self.window_size, self.K*3)
        self.rnn1 = nn.GRU(input_size=self.vocab_size, hidden_size=hidden_size)
        #self.I = nn.Linear(self.vocab_size, self.hidden_size)
        self.rnn2 = nn.GRU(input_size=1+self.vocab_size, hidden_size=hidden_size)
        self.rnn3 = nn.GRU(input_size=self.hidden_size, hidden_size=hidden_size)

        #self.window = AttentionKGaussianWindow(K=K, hidden_size=hidden_size, device=device, version=version)
        #self.Wh1y = nn.Linear(hidden_size, out_size)
        #self.Wh2y = nn.Linear(hidden_size, out_size)
        #self.Wh3y = nn.Linear(hidden_size, out_size)
        # self.by = nn.Parameter(torch.zeros(out_size)).to(device)  # self.by.to(device) not works, Linear includes bias term
        # self.w_to_out = nn.Linear(self.window_size, out_size)
        self.Y = nn.Linear(self.hidden_size, self.n_class)
        self.mse_loss = nn.MSELoss()

        self.version = version
        # optional units
        self.train_init_params = True

        # trainable initial parameters
        if self.train_init_params:
            self.h1_0 = nn.Parameter(torch.zeros((hidden_size)))
            self.h2_0 = nn.Parameter(torch.zeros((hidden_size)))
            self.h3_0 = nn.Parameter(torch.zeros((hidden_size)))
            self.w_0 = nn.Parameter(torch.zeros((self.vocab_size)))

        self.relu = nn.ReLU()

        # logsoftmax
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # criterion
        self.criterion = nn.CrossEntropyLoss()


        # send weights to device
        self.to_device(device)

        # print summary
        self.print_summary()

    def to_device(self, device=None):
        # if device.startswith('cpu'):
        #    self.__device__ = device
        # elif not device.startswith('cuda'):
        assert device and (device.startswith('cuda') or device.startswith('cpu')), f'invalid device {device}'
        device = torch.device(device)
        #self.window.to(device)
        #self.H1.to(device), self.H2.to(device), self.H3.to(device)
        self.rnn1.to(device), self.rnn2.to(device), self.rnn3.to(device)
        #self.Wh1y.to(device), self.Wh2y.to(device), self.Wh3y.to(device)
        self.Wh1p.to(device)
        #self.I.to(device)
        self.Y.to(device),
        self.mse_loss.to(device)
        if self.train_init_params:
            self.h1_0.to(device), self.h2_0.to(device), self.h3_0.to(device), self.w_0.to(device)
        # self.w_to_out.to(device), self.h_to_y.to(device)

    def initial_states(self, batch_size):
        if self.train_init_params:
            h2_0, h3_0, w_0 = \
                self.h2_0.repeat(1, 1, batch_size).view(1, batch_size, -1).to(self.__device__), \
                self.h3_0.repeat(1, 1, batch_size).view(1, batch_size, -1).to(self.__device__), \
                self.w_0.repeat(1, 1, batch_size).view(batch_size, -1).to(self.__device__)
            kappa_0 = torch.zeros((batch_size, self.K)).to(self.__device__)
        else:
            h2_0 = torch.zeros((1, batch_size, self.hidden_size)).to(self.__device__)
            h3_0 = torch.zeros((1, batch_size, self.hidden_size)).to(self.__device__)
            kappa_0 = torch.zeros((batch_size, self.K)).to(self.__device__)
            w_0 = torch.zeros(( batch_size, self.vocab_size)).to(self.__device__)  # TODO: window size
        return h2_0, h3_0, kappa_0, w_0

    def forward(
            self, xt, cx, encoder_hidden, wt_1=None, h2t_1=None, h3t_1=None, kappa_t_1=None):
        #import pdb; pdb.set_trace()

        N = xt.shape[0]
        U = cx.shape[1]
        V = cx.shape[2]

        if wt_1 is None:
            h2t_1, h3, kappa_t_1, wt_1 = self.initial_states(N)

        # encoder_hidden (T=1, N, hidden_size), wt_1 (T=1, N, V)
        alpha_beta_kappa = self.Wh1p(torch.cat([encoder_hidden.squeeze(), wt_1], dim=1))
        K = self.K
        alpha_t, beta_t, kappa_t = alpha_beta_kappa[:, :K], \
                                   alpha_beta_kappa[:, K:K * 2], \
                                   alpha_beta_kappa[:, K * 2:K * 3]

        # αt = exp(αhat_t)  # importance of the window
        # βt = exp(βhat_t)  # width of the window
        # κt = κt-1 + exp(κhat_t)  # location of the window (how far to slide each window)
        alpha_t = alpha_t.exp() + self.epsilon
        beta_t = beta_t.exp() + self.epsilon
        kappa_t = kappa_t_1 + kappa_t.exp()

        # φ(t, u) = Σk=1->K αkt*exp(-βkt(κkt-u)^2)  # mixture of K Gaussian
        u = torch.Tensor(range(cx.shape[1])).to(torch.device(self.__device__))
        phi_t = (alpha_t.unsqueeze(2) *
                 (-beta_t.unsqueeze(2) * (
                    kappa_t.unsqueeze(2).repeat(1, 1, cx.shape[1]) - u) ** 2).exp()
                 ).sum(dim=1)

        wt = (cx.float() * phi_t.unsqueeze(2)).sum(dim=1)

        # decode to signal
        #out = self.I(w)
        #res = out
        out, h2t = self.rnn2(torch.cat([xt.unsqueeze(-1), wt], dim=1).unsqueeze(0), h2t_1)
        #res = out + res
        res = out
        out, h3t = self.rnn3(self.relu(out), h3t_1)
        out = out + res
        dist = self.Y(self.relu(out)).transpose(0, 1)  # (T, N, H(hidden_size)) to (N, T, H(n_class))

        return dist, h2t, h3t, phi_t, wt, kappa_t

    def forward2(
            self, x, cx, encoder_hidden):
        #import pdb; pdb.set_trace()

        N = x.shape[0]
        T = x.shape[1]
        U = cx.shape[1]
        V = cx.shape[2]

        h2, h3, kappa_t_1, wt_1 = self.initial_states(N)

        # attn window
        w = []
        phi = []
        for t in range(T):
            # encoder_hidden (T=1, N, hidden_size), wt_1 (T=1, N, V)
            alpha_beta_kappa = self.Wh1p(torch.cat([encoder_hidden.squeeze(), wt_1], dim=1))
            K = self.K
            alpha_t, beta_t, kappa_t = alpha_beta_kappa[:, :K], \
                                       alpha_beta_kappa[:, K:K * 2], \
                                       alpha_beta_kappa[:, K * 2:K * 3]

            # αt = exp(αhat_t)  # importance of the window
            # βt = exp(βhat_t)  # width of the window
            # κt = κt-1 + exp(κhat_t)  # location of the window (how far to slide each window)
            alpha_t = alpha_t.exp() + self.epsilon
            beta_t = beta_t.exp() + self.epsilon
            kappa_t = kappa_t_1 + kappa_t.exp()

            # φ(t, u) = Σk=1->K αkt*exp(-βkt(κkt-u)^2)  # mixture of K Gaussian
            u = torch.Tensor(range(cx.shape[1])).to(torch.device(self.__device__))
            phi_t = (alpha_t.unsqueeze(2) *
                     (-beta_t.unsqueeze(2) * (
                        kappa_t.unsqueeze(2).repeat(1, 1, cx.shape[1]) - u) ** 2).exp()
                     ).sum(dim=1)

            wt = (cx.float() * phi_t.unsqueeze(2)).sum(dim=1)

            w += [wt]
            phi += [phi_t]

            wt_1 = wt

        #import pdb; pdb.set_trace()

        w = torch.stack(w, dim=0)
        phi = torch.stack(phi, dim=0)

        # decode to signal
        #out = self.I(w)
        #res = out
        out, h2 = self.rnn2(self.relu(w), h2)
        #res = out + res
        res = out
        out, h3 = self.rnn3(self.relu(out), h3)
        out = out + res
        dist = self.Y(self.relu(out)).transpose(0, 1)  # (T, N, H(hidden_size)) to (N, T, H(n_class))

        return dist, h2, h3, phi, w

    def calculate_loss(self, dist, yt):
        #import pdb; pdb.set_trace()
        # dist: (N, n_class), yt.unsqueeze(1): (N)
        return self.criterion(dist.squeeze(1), yt.long())

    #def generate(self, phonemes: list, duration_model: PhonemeDuration, sample_rate=24000):
    #    """Generate Sequence.

    #    args:
    #        - phonemes (list): list of Phoneme43 index
    #        - duration_model (DurationModel):
    #    """
    #    # import pdb; pdb.set_trace()

    #    with torch.no_grad():
    #        # onehot encoding
    #        c = torch.LongTensor(
    #            to_onehot(phonemes, n_class=self.vocab_size)).unsqueeze(0).to(self.__device__)

    #        # batch size is always 1
    #        N = 1
    #        T = int((sum(duration_model.predict(phonemes)[1]) / 1000) * sample_rate // 299.19)

    #        # initial states
    #        h1t_1, h2t_1, h3t_1, kappa_t_1, wt_1 = self.initial_states(N)
    #        x = torch.zeros((N, self.feature_size)).to(self.__device__)

    #        features = []
    #        phi = []
    #        w = []
    #        for t in range(T):
    #            h1t = self.H1(torch.cat([x, wt_1], dim=1), h1t_1)
    #            wt, kappa_t, phi_t = self.window.forward(h1t, kappa_t_1, c)
    #            h2t = self.H2(torch.cat([x, wt, h1t], dim=1), h2t_1)
    #            h3t = self.H3(torch.cat([x, wt, h2t], dim=1), h3t_1)
    #            h = self.Wh1y(h1t) + self.Wh2y(h2t) + self.Wh3y(h3t)  # + self.w_to_out(wt)
    #            y = self.Y(h)

    #            # update previous states
    #            h1t_1, h2t_1, h3t_1, wt_1 = h1t, h2t, h3t, wt
    #            x = y

    #            features += [y]
    #            phi += [phi_t]
    #            w += [wt]

    #        if not self.__device__.startswith('cpu'):
    #            return torch.stack(features, dim=0).squeeze().cpu().detach().numpy(), \
    #                   torch.stack(phi, dim=0).squeeze().cpu().detach().numpy(), \
    #                   torch.stack(w, dim=0).squeeze().cpu().detach().numpy()
    #        else:
    #            return torch.stack(features, dim=0).squeeze().detach().numpy(), \
    #                   torch.stack(phi, dim=0).squeeze().detach().numpy(), \
    #                   torch.stack(w, dim=0).squeeze().detach().numpy()


    def print_summary(self):
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
        settings = {k: v for k, v in settings.items() if k in setting_params}
        assert all([key in settings.keys() for key in setting_params])
        model = cls(**{**settings, **kwargs, **{'device': device}})
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