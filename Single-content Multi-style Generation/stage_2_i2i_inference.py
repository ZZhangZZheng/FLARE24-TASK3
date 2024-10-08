import os
import torch
from sklearn.cluster import KMeans
import numpy as np
from i2i_solver import i2iSolver
import random
import argparse
import matplotlib.pyplot as plt
import nibabel as nib

parser = argparse.ArgumentParser()
# parser.add_argument('--ckpt_path', type=str, default='/home/bd/project/zZ/DAR-UNet/version11_from2080/enc_0040.pt')
parser.add_argument('--ckpt_path', type=str, default='/home/bd/project/zZ/DAR-UNet/version11/i2i_checkpoints/enc_0040.pt')
parser.add_argument('--source_npy_dirpath', type=str, default='/home/bd/project/Flare/Training/FLARE_LabeledCase50/images')
parser.add_argument('--target_npy_dirpath', type=str, default='/home/bd/project/Flare/FLARE24-Task3-MR/Training/LLD-MMRI-3984')
parser.add_argument('--save_npy_dirpath', type=str, default='/home/bd/project/zZ/DAR_UNet_Flare/source2target_training_npy')
parser.add_argument('--k_means_clusters', type=int, default=8)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
opts = parser.parse_args()
trainer=i2iSolver(None)
state_dict = torch.load(opts.ckpt_path)
trainer.enc_c.load_state_dict(state_dict['enc_c'])
trainer.enc_s_a.load_state_dict(state_dict['enc_s_a'])
trainer.enc_s_b.load_state_dict(state_dict['enc_s_b'])
trainer.dec.load_state_dict(state_dict['dec'])
trainer.cuda()

styles=[]
if not os.path.exists('/home/bd/project/zZ/DAR-UNet/source2target_training'):
    os.makedirs('/home/bd/project/zZ/DAR-UNet/source2target_training')
    print(1)

for f2 in os.listdir(opts.target_npy_dirpath):
    if 'label' not in f2:
        img = nib.load(os.path.join(opts.target_npy_dirpath, f2))
        imgs = img.get_fdata().transpose((1, 0, 2))[::-1]
        print(f2,imgs.shape)
        for i in range(int(imgs.shape[-1]/8),int(imgs.shape[-1]/8*7)):
            img = imgs[:, :, i]
            with torch.no_grad():
                single_img = torch.from_numpy((img * 2 - 1)).unsqueeze(0).unsqueeze(0).cuda().float()
                s=trainer.enc_s_b(single_img).cpu().numpy()[0]
                styles.append(s)
n_clusters=opts.k_means_clusters
k_mean_results = KMeans(n_clusters=opts.k_means_clusters, random_state=9).fit_predict(styles)

e=0

for f in os.listdir(opts.source_npy_dirpath):
    collections=[]
    if 'label' not in f:
        img = nib.load(os.path.join(opts.source_npy_dirpath, f))
        imgs = img.get_fdata().transpose((1, 0, 2))[::-1]
        affine=img.affine
        print(f,imgs.shape)
        for k in range(n_clusters):
            nimgs = np.zeros_like(imgs, dtype=np.float32)
            idx = random.choice(np.argwhere(k_mean_results == k).flatten().tolist())
            s = torch.from_numpy(styles[idx]).unsqueeze(0).cuda().float()
            for i in range(imgs.shape[-1]):
                img = imgs[:, :, i]
                single_img = torch.from_numpy((img * 2 - 1)).unsqueeze(0).unsqueeze(0).cuda().float()
                transfered_img = trainer.inference(single_img, s)
                transfered_img = (((transfered_img + 1) / 2).cpu().numpy()).astype(np.float32)[0, 0]
                nimgs[:, :, i] = transfered_img
            label = nib.load(os.path.join('/home/bd/project/Flare/Training/FLARE_LabeledCase50/labels',
                                           f.replace('_0000.nii.gz', '.nii.gz')))
            nlabels=label.get_fdata().transpose((1, 0, 2))[::-1]
            nii_nimgs=nib.Nifti1Image(nimgs,affine=affine)
            nii_nlabels = nib.Nifti1Image(nlabels, affine=affine)
            nib.save(nii_nimgs,'/home/bd/project/zZ/Flare2022-NnUnet/nnUNet_raw_data_base/nnUNet_raw/nnUNet_raw_data/Task022_FLARE22/imagesTr'+'/'+f.replace('_0000', f'_{k}_0000'))
            nib.save(nii_nlabels, '/home/bd/project/zZ/Flare2022-NnUnet/nnUNet_raw_data_base/nnUNet_raw/nnUNet_raw_data/Task022_FLARE22/labelsTr'+'/'+f.replace('_0000', f'_{k}'))
            # np.save(os.path.join('/home/bd/project/zZ/DAR_UNet_Flare/source2target_training_npy',_
            #                      f.replace('image', f'{k}_image')), nimgs)
            # np.save(os.path.join('/home/bd/project/zZ/DAR_UNet_Flare/source2target_training_npy',
            #                      f.replace('image', f'{k}_label')), nlabels)
            collections.append(nimgs[:, :, imgs.shape[-1] // 2])
            collections.append(nlabels[:, :, imgs.shape[-1] // 2])
        for k in range(2):
            for j in range(8):
                plt.subplot(2, 8, k * 8 + j + 1)
                plt.imshow(collections[k * 8 + j], cmap='gray')
                plt.axis('off')
                plt.title(f' {k * 8 + j + 1}')
        plt.tight_layout()

        plt.savefig(f'/home/bd/project/zZ/DAR-UNet/png/{e}.png', dpi=200)
        print(e, f)
        e = e + 1
        plt.close()





