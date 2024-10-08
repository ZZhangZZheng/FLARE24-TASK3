from models import ContentEncoder,Decoder,MsImageDis,content_Dis,StyleEncoder
from adabelief_pytorch import AdaBelief
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import itertools
import time
import os
import copy
from torchsummary import summary

class i2iSolver(nn.Module):
    def __init__(self, opts, TTUR=True):
        super().__init__()
        self.opts=opts

        self.enc_c=ContentEncoder()
        self.enc_s_a=StyleEncoder(style_dim=8)
        self.enc_s_b=StyleEncoder(style_dim=8)
        self.dec=Decoder(style_dim=8)

        self.enc_c_ema=copy.deepcopy(self.enc_c)
        self.enc_s_a_ema=copy.deepcopy(self.enc_s_a)
        self.enc_s_b_ema=copy.deepcopy(self.enc_s_b)
        self.dec_ema=copy.deepcopy(self.dec)

        self.dis_a=MsImageDis()
        self.dis_b=MsImageDis()
        self.dis_c=content_Dis()

        self.gen_opt = AdaBelief(itertools.chain(self.enc_c.parameters(), self.enc_s_a.parameters(),self.enc_s_b.parameters(),self.dec.parameters()), lr=1e-4, weight_decay=0,eps=1e-16, betas=(0.5, 0.9), weight_decouple=True, rectify=True, print_change_log=False)
        self.dis_opt = AdaBelief(itertools.chain(self.dis_a.parameters(), self.dis_b.parameters(), self.dis_c.parameters()), lr=2e-4, weight_decay=0, eps=1e-16, betas=(0.5, 0.9), weight_decouple=True, rectify=True, print_change_log=False)

        self.recon_criterion=nn.L1Loss()

        self.lambda_rec=10
        self.lambda_cyc=10
        self.lambda_fm=1

        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def inference(self,x,r):
        with torch.no_grad():
            self.x = self.dec(self.enc_c(x), r)
        return self.x


    def gan_forward(self,x_a,x_b):
        self.x_a = x_a
        self.x_b = x_b

        self.s_a = self.enc_s_a(self.x_a)
        self.s_b = self.enc_s_b(self.x_b)

        self.c_a = self.enc_c(self.x_a)
        self.c_b = self.enc_c(self.x_b)

        self.x_a_recon = self.dec(self.c_a, self.s_a)
        self.x_b_recon = self.dec(self.c_b, self.s_b)

        self.x_ab = self.dec(self.c_a, self.s_b)
        self.x_ba = self.dec(self.c_b, self.s_a)

        self.c_a_recon = self.enc_c(self.x_ab)
        self.c_b_recon = self.enc_c(self.x_ba)

        self.s_b_recon = self.enc_s_b(self.x_ab)
        self.s_a_recon = self.enc_s_a(self.x_ba)

        self.x_aba = self.dec(self.c_a_recon, self.s_a)
        self.x_bab = self.dec(self.c_b_recon, self.s_b)

    def forward(self, x_a):
        self.x_a = x_a
        self.x_b = x_a

        self.s_a = self.enc_s_a(self.x_a)
        self.s_b = self.enc_s_b(self.x_b)

        self.c_a = self.enc_c(self.x_a)
        self.c_b = self.enc_c(self.x_b)

        self.x_a_recon = self.dec(self.c_a, self.s_a)
        self.x_b_recon = self.dec(self.c_b, self.s_b)

        self.x_ab = self.dec(self.c_a, self.s_b)
        self.x_ba = self.dec(self.c_b, self.s_a)

        self.c_a_recon = self.enc_c(self.x_ab)
        self.c_b_recon = self.enc_c(self.x_ba)

        self.s_b_recon = self.enc_s_b(self.x_ab)
        self.s_a_recon = self.enc_s_a(self.x_ba)

        self.x_aba = self.dec(self.c_a_recon, self.s_a)
        self.x_bab = self.dec(self.c_b_recon, self.s_b)

    def gen_update(self):
        self.gen_opt.zero_grad()

        self.loss_g_rec = (self.recon_criterion(self.x_a, self.x_a_recon) + self.recon_criterion(self.x_b, self.x_b_recon)) * self.lambda_rec
        self.loss_g_cyc = (self.recon_criterion(self.x_a, self.x_aba) + self.recon_criterion(self.x_b, self.x_bab)) * self.lambda_cyc
        self.loss_g_fm = (self.recon_criterion(self.c_a, self.c_a_recon) + self.recon_criterion(self.c_b, self.c_b_recon)) *self.lambda_fm
        self.loss_g_rec_s = (self.recon_criterion(self.s_a, self.s_a_recon) + self.recon_criterion(self.s_b, self.s_b_recon)) * 1
        self.loss_g_adv = self.dis_a.calc_gen_loss(self.x_ba,self.x_a) + self.dis_b.calc_gen_loss(self.x_ab,self.x_b)
        self.loss_c_adv=self.dis_c.calc_gen_loss(self.c_b,self.c_a)

        self.loss_g = self.loss_g_cyc + self.loss_g_fm + self.loss_g_adv + self.loss_g_rec + self.loss_c_adv+ self.loss_g_rec_s
        self.loss_g.backward()
        self.gen_opt.step()

        self.moving_average(self.enc_c, self.enc_c_ema, beta=0.999)
        self.moving_average(self.enc_s_a, self.enc_s_a_ema, beta=0.999)
        self.moving_average(self.enc_s_b, self.enc_s_b_ema, beta=0.999)
        self.moving_average(self.dec, self.dec_ema, beta=0.999)

    def dis_update(self):
        self.dis_opt.zero_grad()
        self.loss_dis_a = self.dis_a.calc_dis_loss(self.x_ba.detach(), self.x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(self.x_ab.detach(), self.x_b)
        self.loss_dis_c= self.dis_c.calc_dis_loss(self.c_b.detach(),self.c_a.detach())
        self.loss_dis_total = self.loss_dis_a + self.loss_dis_b+self.loss_dis_c
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def verbose(self):
        text=''
        for s in self.__dict__.keys():
            if 'loss_' in s:
                text+='{} {:.3f}  '.format(s.replace('loss_',''),getattr(self,s).item())
        return text

    def gan_visual(self,epoch):
        collections=[]
        for im in [self.x_a, self.x_a_recon, self.x_ab, self.x_aba, self.x_b,self.x_b_recon,self.x_ba, self.x_bab]:
            tim= np.clip(((im[0,0].detach().cpu().numpy())+1)*127.5,0,255).astype(np.uint8)
            collections.append(tim)
        for i in range(2):
            for j in range(4):
                plt.subplot(2,4,i*4+j+1)
                plt.imshow(collections[i*4+j],cmap='gray')
                plt.axis('off')
        plt.tight_layout()
        e='%03d'%epoch
        plt.savefig(f'{self.opts.name}/i2i_train_visual/{e}_{time.time()}.png',dpi=200)
        plt.close()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def moving_average(self, model, model_test, beta=0.999):
        for param, param_test in zip(model.parameters(), model_test.parameters()):
            param_test.data = torch.lerp(param.data, param_test.data, beta)

    def save(self,  epoch):
        model_name = os.path.join(self.opts.name,'i2i_checkpoints', 'enc_%04d.pt' % (epoch + 1))
        torch.save({'enc_c': self.enc_c_ema.state_dict(), 'dec': self.dec_ema.state_dict(),
                    'enc_s_a': self.enc_s_a_ema.state_dict(),'enc_s_b': self.enc_s_b_ema.state_dict(),
                    'dis_a': self.dis_a.state_dict(), 'dis_b': self.dis_b.state_dict()}, model_name)


# ####计算参数量################################################################################
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--name', type=str, default='version11')
# parser.add_argument('--seed', type=int, default=10)
# opts = parser.parse_args()
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = i2iSolver(opts).to(device)
# summary(model, (1, 256, 256))
# # summary(model, [(1, 256, 256),(1, 256, 256)])

#'enc_ema': self.enc_ema.state_dict(), 'dec_ema': self.dec_ema.state_dict(),


############计算flops##################################################################
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--name', type=str, default='version11')
# parser.add_argument('--seed', type=int, default=10)
# opts = parser.parse_args()
# from fvcore.nn import FlopCountAnalysis
# from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table
# from utils import I2IDataset,create_dirs
# from torch.utils.data import DataLoader
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# model=i2iSolver(opts)
# model.cuda()
# train_loader = DataLoader(dataset=I2IDataset(train=True), batch_size=6, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
# validation_loader = DataLoader(dataset=I2IDataset(train=False), batch_size=1, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
# for train_data in train_loader:
#     for k in train_data.keys():
#         train_data[k] = train_data[k].cuda()#.detach()
#     input=train_data['B_img'].cuda()
#     flop = FlopCountAnalysis(model, input)
#     print(flop_count_table(flop,  show_param_shapes=False))
#     print(flop_count_str(flop))
#     print("Total", flop.total() / 1e9)
#     break
#     #########A: (2939, 512, 512)
# ####################计算FLOPS###########################       https://github.com/sovrasov/flops-counter.pytorch#
# from ptflops import get_model_complexity_info
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--name', type=str, default='version11')
# parser.add_argument('--seed', type=int, default=10)
# opts = parser.parse_args()
# from fvcore.nn import FlopCountAnalysis
# from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table
# from utils import I2IDataset,create_dirs
# from torch.utils.data import DataLoader
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# model=i2iSolver(opts)
# with torch.cuda.device(0):
#     macs, params = get_model_complexity_info(model, (1, 256, 256), as_strings=True, backend='pytorch',print_per_layer_stat = True, verbose = True)
#     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
#     #######Computational complexity:       100.36 GMac
#     #########Number of parameters:           24.4 M

