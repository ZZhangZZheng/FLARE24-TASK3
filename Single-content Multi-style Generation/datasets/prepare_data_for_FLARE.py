
import nibabel as nib
import glob
import matplotlib.pyplot as plt
import pydicom
import os

import numpy as np

import SimpleITK as sitk
import torch
import torch.nn.functional as F
#填充至512*512
def pad(im):
    shape=im.shape
    #cx,cy,cz=int((bbox[0]+bbox[3])/2),int((bbox[1]+bbox[4])/2),int((bbox[2]+bbox[5])/2)
    cx,cy= int((shape[0])/2),int(shape[1]/2)
    shape=im.shape
    ss=[[256-cx,256+shape[0]-cx],[256-cy,256+shape[1]-cy]]
    empty1=np.zeros([512,512,im.shape[-1]])
    empty1[ss[0][0]:ss[0][1],ss[1][0]:ss[1][1]]=im
    return empty1

for dir in ['source_training_npy','target_training_npy','source_test_npy','target_test_npy']:
    if not os.path.isdir(dir):
        os.makedirs(dir)

def ct_npy():
    # 1-50
    item = 1
    # root=r'D:/Dataset/FLARE/Training/FLARE_LabeledCase50/images'
    imgs_path = sorted(glob.glob("/mnt/sda1/project/zZ/Flare/D:/Flare/Training/FLARE_LabeledCase50/images/*.nii.gz"))
    # FLARE22_Tr_0001_0000.nii.gz 512*512*110
    A_imgs = []
    UID = item
    for f in imgs_path:
        img = nib.load(f)

        image_data = img.get_fdata().transpose((1, 0, 2))[::-1]

        print(f, UID, img.shape)

        mask_path = f.replace('images', 'labels').replace("_0000.nii.gz", ".nii.gz")
        print(mask_path)
        mask_data = nib.load(mask_path).get_fdata().transpose((1, 0, 2))[::-1]
        seg = sitk.ReadImage(mask_path)
        seg_array = sitk.GetArrayFromImage(seg)
        # 遍历数组中的值，将值为0、1、2、3和6的元素保持不变，其他元素设置为0。
        # for i in range(14):
        #     if i in [0,1,2,3,6]:
        #         pass
        #     else:
        #         seg_array[seg_array==i]=0
        # 根据0进行裁剪
        z = np.any(seg_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]
        start_slice = max(start_slice - 2, 0)
        end_slice = min(end_slice + 2, image_data.shape[-1])
        image_data = image_data[:, :, start_slice:end_slice + 1]
        mask_data = mask_data[:, :, start_slice:end_slice + 1]

        # 归一化（0-1）
        image_data[image_data > 350] = 350
        image_data[image_data < -350] = -350
        image_data = (np.clip(image_data, -350, 350) + 350) / 700
        origin_shape = image_data.shape
        target_shape = [int(origin_shape[0] * img.header['pixdim'][1]), int(origin_shape[1] * img.header['pixdim'][2]),
                        int(origin_shape[2] * img.header['pixdim'][3] / 4)]

        image_data = torch.from_numpy(image_data).unsqueeze(0).unsqueeze(0)
        image_data = F.interpolate(image_data, target_shape, mode="trilinear").numpy()[0, 0]
        image_data = pad(image_data)
        print("target_shape:", target_shape)
        # print(image_data.shape,mask_data.shape,mask_data.dtype,np.unique(mask_data))

        mask_data = torch.from_numpy(mask_data.copy()).unsqueeze(0).unsqueeze(0)
        mask_data = F.interpolate(mask_data, target_shape, mode='nearest').numpy()[0, 0]
        mask_data = pad(mask_data)

        # 改标签但我们不需要
        # old_mask_data=mask_data
        # mask_data=np.zeros_like(mask_data)
        # mask_data[old_mask_data==6]=1
        # mask_data[old_mask_data==2]=2
        # mask_data[old_mask_data == 3] = 3
        # mask_data[old_mask_data == 1] = 4

        print("img:", image_data.shape, image_data.dtype, image_data.max(), image_data.min(), img.header['pixdim'])
        print("mask:", np.unique(mask_data))
        # plt.imshow(image_data[:,:,24],'gray',vmin=0,vmax=1)
        # plt.show()
        # if UID  in [5,10,15,20,25,30,35,40,45,50]:
        A_imgs.append(image_data)


        np.save('/mnt/sda1/project/zZ/Flare/D:/DAR-Unet_Flare/source_training_npy/{}_image.npy'.format(UID), image_data)
        np.save('/mnt/sda1/project/zZ/Flare/D:/DAR-Unet_Flare/source_training_npy/{}_label.npy'.format(UID), mask_data)

        UID = UID + 1
    A_imgs = np.concatenate(A_imgs, -1).transpose((2, 0, 1))
    np.save('A_imgs.npy', A_imgs)

from PIL import Image


'''
The data sets are acquired by a 1.5T Philips MRI, 
which produces 12 bit DICOM images having a resolution of 256 x 256. 
The ISDs vary between 5.5-9 mm (average 7.84 mm), 
x-y spacing is between 1.36 - 1.89 mm (average 1.61 mm) 
and the number of slices is between 26 and 50 (average 36). 
In total, 1594 slices (532 slice per sequence) will be provided for training and 1537 slices will be used for the tests.
'''

def mri_npy():
    B_imgs = []
    # root=r'D:\PycharmProjects\visualization\heartseg\abdomen_dataset_nii\abdomen_MR_nii'
    imgs_path = sorted(glob.glob("/mnt/sda1/project/zZ/Flare/D:/Flare/FLARE24-Task3-MR/Training/AMOS_MR_good_spacing-833/*.nii.gz"))
    item = 3000
    UID = item
    for d in imgs_path:  #
        img = nib.load(d)

        image_data = img.get_fdata().transpose((1, 0, 2))[::-1]

        print(d, UID, img.shape)

        origin_shape = image_data.shape
        target_shape = [int(origin_shape[0] * img.header['pixdim'][1]), int(origin_shape[1] * img.header['pixdim'][2]),
                        int(origin_shape[2] * img.header['pixdim'][3] / 4)]

        # image_data=np.stack(im_list,-1).astype(np.float32)
        image_data /= image_data.max()
        origin_shape = image_data.shape
        # mask_data=np.stack(msk_list,-1)

        # target_shape= [int(origin_shape[0]*pixdim[0]),int(origin_shape[1]*pixdim[1]),origin_shape[2]*2]
        image_data = torch.from_numpy(image_data.copy()).unsqueeze(0).unsqueeze(0)
        image_data = F.interpolate(image_data, target_shape, mode='trilinear').numpy()[0, 0]
        image_data = pad(image_data)

        # mask_data = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0)
        # mask_data = F.interpolate(mask_data, target_shape, mode='nearest').numpy()[0, 0]
        # mask_data = pad(mask_data)

        print("target_shape:", target_shape, "image_shape:", image_data.shape)

        np.save('/mnt/sda1/project/zZ/Flare/D:/DAR-Unet_Flare/target_training_npy/{}_image.npy'.format(UID), image_data)
        if UID in [3000,3050,3100, 3150,3200, 3250,3300, 3350,3400]:
            B_imgs.append(image_data)
        # if int(d) not in [1,13,32,38]:
        #     np.save(f'target_training_npy/{d}_img.npy',image_data)
        #     # np.save(f'target_training_npy/{d}_label.npy',mask_data)
        #     B_imgs.append(image_data)
        # else:
        #     print(d)
        #     np.save(f'target_test_npy/{d}_img.npy',image_data)
        #     np.save(f'target_test_npy/{d}_label.npy',mask_data)
        UID = UID + 1
        if UID>3500:
            break

    imgs_path = sorted(
        glob.glob("/mnt/sda1/project/zZ/Flare/D:/Flare/FLARE24-Task3-MR/Training/LLD-MMRI-3984/*.nii.gz"))
    item = 4000
    UID = item
    for d in imgs_path:  #
        img = nib.load(d)

        image_data = img.get_fdata().transpose((1, 0, 2))[::-1]

        print(d, UID, img.shape)

        origin_shape = image_data.shape
        target_shape = [int(origin_shape[0] * img.header['pixdim'][1]),
                        int(origin_shape[1] * img.header['pixdim'][2]),
                        int(origin_shape[2] * img.header['pixdim'][3] / 4)]

        # image_data=np.stack(im_list,-1).astype(np.float32)
        image_data /= image_data.max()
        origin_shape = image_data.shape
        # mask_data=np.stack(msk_list,-1)

        # target_shape= [int(origin_shape[0]*pixdim[0]),int(origin_shape[1]*pixdim[1]),origin_shape[2]*2]
        image_data = torch.from_numpy(image_data.copy()).unsqueeze(0).unsqueeze(0)
        image_data = F.interpolate(image_data, target_shape, mode='trilinear').numpy()[0, 0]
        image_data = pad(image_data)

        # mask_data = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0)
        # mask_data = F.interpolate(mask_data, target_shape, mode='nearest').numpy()[0, 0]
        # mask_data = pad(mask_data)

        print("target_shape:", target_shape, "image_shape:", image_data.shape)

        np.save('/mnt/sda1/project/zZ/Flare/D:/DAR-Unet_Flare/target_training_npy/{}_image.npy'.format(UID), image_data)
        if UID % 300 == 0 or (UID-1) % 300 == 0 or (UID-2) % 300 == 0 or (UID-3) % 300 == 0 or (UID-4)% 300 == 0 or (UID-5) % 300 == 0 or (UID-6)%300==0 or (UID-7)%300==0:
            B_imgs.append(image_data)
            print(UID)
        # if int(d) not in [1,13,32,38]:
        #     np.save(f'target_training_npy/{d}_img.npy',image_data)
        #     # np.save(f'target_training_npy/{d}_label.npy',mask_data)
        #     B_imgs.append(image_data)
        # else:
        #     print(d)
        #     np.save(f'target_test_npy/{d}_img.npy',image_data)
        #     np.save(f'target_test_npy/{d}_label.npy',mask_data)
        UID = UID + 1
        if UID>5000:
            break

    B_imgs = np.concatenate(B_imgs, -1).transpose((2, 0, 1))
    np.save('B_imgs.npy', B_imgs)


def test_npy():
    imgs_path = sorted(glob.glob("/mnt/sda1/project/zZ/Flare/D:/Flare/FLARE24-Task3-MR/PublicValidation/imagesVal/*.nii.gz"))
    item = 9000
    UID = item
    for t in imgs_path:  #
        img = nib.load(t)

        image_data = img.get_fdata().transpose((1, 0, 2))[::-1]

        print(t, UID, img.shape)

        mask_path = t.replace('images', 'labels').replace("_0000.nii.gz", ".nii.gz")
        mask_data = nib.load(mask_path).get_fdata().transpose((1, 0, 2))[::-1]

        origin_shape = image_data.shape
        target_shape = [int(origin_shape[0] * img.header['pixdim'][1]), int(origin_shape[1] * img.header['pixdim'][2]),
                        int(origin_shape[2] * img.header['pixdim'][3] / 4)]
        if target_shape[0]>512:
            target_shape[0]=512
        if target_shape[1]>512:
            target_shape[1]=512
        image_data /= image_data.max()
        origin_shape = image_data.shape

        image_data = torch.from_numpy(image_data.copy()).unsqueeze(0).unsqueeze(0)
        image_data = F.interpolate(image_data, target_shape, mode='trilinear').numpy()[0, 0]
        #print(image_data.shape)
        image_data = pad(image_data)


        mask_data = torch.from_numpy(mask_data.copy()).unsqueeze(0).unsqueeze(0)
        mask_data = F.interpolate(mask_data, target_shape, mode='nearest').numpy()[0, 0]
        mask_data = pad(mask_data)

        print("origin_shape:",origin_shape,"target_shape:", target_shape, "image_shape:", image_data.shape)
        # print(image_data.shape,image_data.dtype,image_data.max(),image_data.min(),dcm.PixelSpacing)
        # plt.imshow(image_data[:, :, 12], 'gray')
        # plt.show()

        np.save('/mnt/sda1/project/zZ/Flare/D:/DAR-Unet_Flare/target_test_npy/{}_image.npy'.format(UID), image_data)
        np.save('/mnt/sda1/project/zZ/Flare/D:/DAR-Unet_Flare/target_test_npy/{}_label.npy'.format(UID), mask_data)
        UID = UID + 1

if __name__ == '__main__':
    ct_npy()
    mri_npy()
    test_npy()



