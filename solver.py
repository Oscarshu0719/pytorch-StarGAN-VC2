import ast
from datetime import datetime, timedelta
import librosa
import numpy as np
import os
from pyworld import decode_spectral_envelope, synthesize
import random
from sklearn.preprocessing import LabelBinarizer
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

from data_loader import TestSet
from model import Discriminator, Generator
from preprocess import FRAMES, SAMPLE_RATE, FFTSIZE
from utility import Normalizer, speakers

class Solver(object):
    def __init__(self, data_loader, config):
        self.config = config
        self.data_loader = data_loader
        self.num_spk = config.num_spk
       
        self.lambda_cyc = config.lambda_cyc
        self.lambda_gp = config.lambda_gp
        self.lambda_id = config.lambda_id

        # Training configurations.
        self.data_dir = config.data_dir
        self.test_dir = config.test_dir
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        
        # Test configurations.
        self.test_iters = config.test_iters
        self.trg_speaker = ast.literal_eval(config.trg_speaker)
        self.src_speaker = config.src_speaker

        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.spk_enc = LabelBinarizer().fit(speakers)

        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        self.build_model()
        # Only use tensorboard in train mode.
        if self.use_tensorboard and config.mode == 'train':
            self.build_tensorboard()
    
    def build_model(self):
        self.G = Generator(num_speakers=self.num_spk)
        self.D = Discriminator(num_speakers=self.num_spk)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)
    
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(f"* ({name}) Number of parameters: {num_params}.")

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """
            Decay learning rates of the generator and discriminator.
        """

        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def train(self):
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        start_iters = 0
        if self.resume_iters:
            print(f'Resume at step {self.resume_iters}...')
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)   

        norm = Normalizer()
        data_iter = iter(self.data_loader)

        g_adv_optim = 0            
        g_adv_converge_low = True  # Check which direction `g_adv` is converging (init as low).
        g_rec_optim = 0            
        g_rec_converge_low = True  # Check which direction `g_rec` is converging (init as low).
        g_tot_optim = 0            
        g_tot_converge_low = True  # Check which direction `g_tot` is converging (init as low).

        print('Start training...')
        start_time = datetime.now()

        for i in range(start_iters, self.num_iters):
            try:
                x_real, speaker_idx_org, label_org = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                x_real, speaker_idx_org, label_org = next(data_iter)           

            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]
            speaker_idx_trg = speaker_idx_org[rand_idx]
            
            x_real = x_real.to(self.device)           
            label_org = label_org.to(self.device)             # Original domain one-hot labels.
            label_trg = label_trg.to(self.device)             # Target domain one-hot labels.
            speaker_idx_org = speaker_idx_org.to(self.device) # Original domain labels.
            speaker_idx_trg = speaker_idx_trg.to(self.device) # Target domain labels.

            """
                Discriminator training.
            """
            CELoss = nn.CrossEntropyLoss()

            # Loss: st-adv.
            out_r = self.D(x_real, label_org, label_trg)
            x_fake = self.G(x_real, label_trg)
            out_f = self.D(x_fake.detach(), label_org, label_trg)
            d_loss_adv = F.binary_cross_entropy_with_logits(input=out_f, target=torch.zeros_like(out_f, dtype=torch.float)) + \
                F.binary_cross_entropy_with_logits(input=out_r, target=torch.ones_like(out_r, dtype=torch.float))
           
            # Loss: gp.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src = self.D(x_hat, label_org, label_trg)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Totol loss: st-adv + lambda_gp * gp.
            d_loss = d_loss_adv + self.lambda_gp * d_loss_gp

            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            loss = {}
            loss['D/d_loss_adv'] = d_loss_adv.item()
            loss['D/d_gp'] = d_loss_gp.item()
            loss['D/d_loss'] = d_loss.item()

            """
                Generator training.
            """        
            if (i + 1) % self.n_critic == 0:
                # Loss: st-adv (original-to-target).
                x_fake = self.G(x_real, label_trg)
                g_out_src = self.D(x_fake, label_org, label_trg)
                g_loss_adv = F.binary_cross_entropy_with_logits(input=g_out_src, target=torch.ones_like(g_out_src, dtype=torch.float))
                
                # Loss: cyc (target-to-original).
                x_rec = self.G(x_fake, label_org)
                g_loss_rec = F.l1_loss(x_rec, x_real)

                # Loss: id (original-to-original).
                x_fake_id = self.G(x_real, label_org)
                g_loss_id = F.l1_loss(x_fake_id, x_real)

                # Total loss: st-adv + lambda_cyc * cyc (+ lambda_id * id).
                # Only include Identity mapping before 10k iterations.
                if (i + 1) < 10 ** 4: 
                    g_loss = g_loss_adv \
                             + self.lambda_cyc * g_loss_rec \
                             + self.lambda_id * g_loss_id
                else:
                    g_loss = g_loss_adv + self.lambda_cyc * g_loss_rec

                # Check convergence direction of losses.
                if (i + 1) == 20 * (10 ** 3):  # Update optims at 20k iterations.
                    g_adv_optim = g_loss_adv
                    g_rec_optim = g_loss_rec
                    g_tot_optim = g_loss
                if (i + 1) == 70 * (10 ** 3):  # Check which direction optims have gone over 70k iters.
                    if g_loss_adv > g_adv_optim:
                        g_adv_converge_low = False
                    if g_loss_rec > g_rec_optim:
                        g_rec_converge_low = False
                    if g_loss > g_tot_optim:
                        g_tot_converge_low = False

                    print('* CONVERGE DIRECTION')
                    print(f'adv_loss low: {g_adv_converge_low}')
                    print(f'g_rec_loss los: {g_rec_converge_low}')
                    print(f'g_loss loq: {g_tot_converge_low}')

                # Update loss for checkpoint saving.
                if (i + 1) > 75 * (10 ** 3): 
                    if g_tot_converge_low:
                        if (g_loss_adv < g_adv_optim and abs(g_loss_adv - g_adv_optim) > 0.1) and g_loss_rec < g_rec_optim:
                            self.save_optim_checkpoints('g_adv_rec_optim-G.ckpt', 'g_adv_rec_optim-D.ckpt', 'adv+rec')
                    elif not g_tot_converge_low:
                        if (g_loss_adv > g_adv_optim and abs(g_loss_adv - g_adv_optim) > 0.1) and g_loss_rec < g_rec_optim:
                            self.save_optim_checkpoints('g_adv_rec_optim-G.ckpt', 'g_adv_rec_optim-D.ckpt', 'adv+rec')

                    if g_adv_converge_low:
                        if g_loss_adv < g_adv_optim:
                            g_adv_optim = g_loss_adv
                            self.save_optim_checkpoints('g_adv_optim-G.ckpt', 'g_adv_optim-D.ckpt', 'adv')
                    elif not g_adv_converge_low:
                        if g_loss_adv < g_adv_optim:
                            g_adv_optim = g_loss_adv
                            self.save_optim_checkpoints('g_adv_optim-G.ckpt', 'g_adv_optim-D.ckpt', 'adv')

                    if g_rec_converge_low:
                        if g_loss_rec < g_rec_optim:
                            g_rec_optim = g_loss_rec
                            self.save_optim_checkpoints('g_rec_optim-G.ckpt', 'g_rec_optim-D.ckpt', 'rec')
                    elif not g_rec_converge_low:
                        if g_loss_rec > g_rec_optim:
                            g_rec_optim = g_loss_rec
                            self.save_optim_checkpoints('g_rec_optim-G.ckpt', 'g_rec_optim-D.ckpt', 'rec')

                    if g_tot_converge_low:
                        if g_loss < g_tot_optim:
                            g_tot_optim = g_loss
                            self.save_optim_checkpoints('g_tot_optim-G.ckpt', 'g_tot_optim-D.ckpt', 'tot')
                    elif not g_tot_converge_low:
                        if g_loss > g_tot_optim:
                            g_tot_optim = g_loss
                            self.save_optim_checkpoints('g_tot_optim-G.ckpt', 'g_tot_optim-D.ckpt', 'tot')

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                loss['G/g_loss_adv'] = g_loss_adv.item()
                loss['G/g_loss_rec'] = g_loss_rec.item()
                loss['G/g_loss_id'] = g_loss_id.item()
                loss['G/g_loss'] = g_loss.item()

            # Print training information.
            if (i + 1) % self.log_step == 0:
                et = datetime.now() - start_time
                et = str(et)[: -7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i + 1)

            # Translate fixed images for debugging.
            if (i + 1) % self.sample_step == 0:
                with torch.no_grad():
                    d, speaker = TestSet(self.test_dir).test_data()
                    target = random.choice([x for x in speakers if x != speaker])
                    label_t = self.spk_enc.transform([target])[0]
                    label_t = np.asarray([label_t])

                    for filename, content in d.items():
                        f0 = content['f0']
                        ap = content['ap']
                        sp_norm_pad = self.pad_coded_sp(content['coded_sp_norm'])
                        
                        convert_result = []
                        for start_idx in range(0, sp_norm_pad.shape[1] - FRAMES + 1, FRAMES):
                            one_seg = sp_norm_pad[:, start_idx: start_idx + FRAMES]
                            
                            one_seg = torch.FloatTensor(one_seg).to(self.device)
                            one_seg = one_seg.view(1,1,one_seg.size(0), one_seg.size(1))
                            l = torch.FloatTensor(label_t)
                            one_seg = one_seg.to(self.device)
                            l = l.to(self.device)
                            one_set_return = self.G(one_seg, l).data.cpu().numpy()
                            one_set_return = np.squeeze(one_set_return)
                            one_set_return = norm.backward_process(one_set_return, target)
                            convert_result.append(one_set_return)

                        convert_con = np.concatenate(convert_result, axis=1)
                        convert_con = convert_con[:, 0: content['coded_sp_norm'].shape[1]]
                        contigu = np.ascontiguousarray(convert_con.T, dtype=np.float64)   
                        decoded_sp = decode_spectral_envelope(contigu, SAMPLE_RATE, fft_size=FFTSIZE)
                        f0_converted = norm.pitch_conversion(f0, speaker, target)
                        wav = synthesize(f0_converted, decoded_sp, ap, SAMPLE_RATE)

                        name = f'{speaker}-{target}_iter{i + 1}_{filename}'
                        path = os.path.join(self.sample_dir, name)
                        print(f'[SAVE]: {path}')
                        librosa.output.write_wav(path, wav, SAMPLE_RATE)
                        
            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print(f'Save model checkpoints into {self.model_save_dir}...')

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print (f'Decayed learning rates, g_lr: {g_lr}, d_lr: {d_lr}.')

    def gradient_penalty(self, y, x):
        """
            Compute gradient penalty: (L2_norm(dy / dx) - 1) ** 2.
            (Differs from the paper.)
        """

        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def save_optim_checkpoints(self, g_name, d_name, type_saving):
        G_path = os.path.join(self.model_save_dir, g_name)
        D_path = os.path.join(self.model_save_dir, d_name)
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        print(f'Save {type_saving} optimal model checkpoints into {self.model_save_dir}...')

    def restore_model(self, resume_iters):
        print(f'Loading the trained models from step {resume_iters}...')
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    @staticmethod
    def pad_coded_sp(coded_sp_norm):
        f_len = coded_sp_norm.shape[1]
        if  f_len >= FRAMES: 
            pad_length = FRAMES-(f_len - (f_len//FRAMES) * FRAMES)
        elif f_len < FRAMES:
            pad_length = FRAMES - f_len

        sp_norm_pad = np.hstack((coded_sp_norm, np.zeros((coded_sp_norm.shape[0], pad_length))))
        return sp_norm_pad 

    def convert(self):
        """
            Convertion.    
        """

        self.restore_model(self.test_iters)
        norm = Normalizer()

        d, speaker = TestSet(self.test_dir).test_data(self.src_speaker)
        targets = self.trg_speaker
       
        for target in targets:
            print(target)
            assert target in speakers
            label_t = self.spk_enc.transform([target])[0]
            label_t = np.asarray([label_t])
            
            with torch.no_grad():
                for filename, content in d.items():
                    f0 = content['f0']
                    ap = content['ap']
                    sp_norm_pad = self.pad_coded_sp(content['coded_sp_norm'])

                    convert_result = []
                    for start_idx in range(0, sp_norm_pad.shape[1] - FRAMES + 1, FRAMES):
                        one_seg = sp_norm_pad[:, start_idx: start_idx + FRAMES]
                        
                        one_seg = torch.FloatTensor(one_seg).to(self.device)
                        one_seg = one_seg.view(1, 1, one_seg.size(0), one_seg.size(1))
                        l = torch.FloatTensor(label_t)
                        one_seg = one_seg.to(self.device)
                        l = l.to(self.device)
                        one_set_return = self.G(one_seg, l).data.cpu().numpy()
                        one_set_return = np.squeeze(one_set_return)
                        one_set_return = norm.backward_process(one_set_return, target)
                        convert_result.append(one_set_return)

                    convert_con = np.concatenate(convert_result, axis=1)
                    convert_con = convert_con[:, 0: content['coded_sp_norm'].shape[1]]
                    contigu = np.ascontiguousarray(convert_con.T, dtype=np.float64)   
                    decoded_sp = decode_spectral_envelope(contigu, SAMPLE_RATE, fft_size=FFTSIZE)
                    f0_converted = norm.pitch_conversion(f0, speaker, target)
                    wav = synthesize(f0_converted, decoded_sp, ap, SAMPLE_RATE)

                    name = f'{speaker}-{target}_iter{self.test_iters}_{filename}'
                    path = os.path.join(self.result_dir, name)
                    print(f'[SAVE]: {path}')
                    librosa.output.write_wav(path, wav, SAMPLE_RATE)            
