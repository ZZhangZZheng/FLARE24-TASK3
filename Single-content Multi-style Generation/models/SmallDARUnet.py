import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, anisotropy_stride=False, from_image=False, anisotropy_dim=0,max=False):
        super(ResBlock, self).__init__()
        #print(output_dim)
        stride=[stride,stride,stride]
        maxstride = [2, 2, 2]
        maxpadding= [1,1,1]
        if anisotropy_stride:
            maxstride[anisotropy_dim]=1
            maxpadding[anisotropy_dim]=0



        if anisotropy_stride:
            stride[anisotropy_dim]=1


        if max:
            if from_image:
                self.conv_block = nn.Sequential(

                    nn.Conv3d(input_dim, output_dim, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1)),
                    nn.GroupNorm(output_dim // 8, output_dim, affine=True),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(output_dim, output_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                    nn.GroupNorm(output_dim // 8, output_dim, affine=True),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(output_dim, output_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                    nn.ConvTranspose3d(output_dim, output_dim, kernel_size=3, stride=maxstride, padding=1,
                                       output_padding=maxpadding)
                )
            else:
                self.conv_block = nn.Sequential(

                    nn.GroupNorm(input_dim // 8, input_dim, affine=True),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(input_dim, output_dim, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1)),

                    nn.GroupNorm(output_dim // 8, output_dim, affine=True),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(output_dim, output_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),

                    nn.GroupNorm(output_dim // 8, output_dim, affine=True),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(output_dim, output_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                    nn.ConvTranspose3d(output_dim, output_dim, kernel_size=3, stride=maxstride, padding=1,
                                       output_padding=maxpadding)
                )

            self.conv_skip = nn.Sequential(

                nn.Conv3d(input_dim, output_dim, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1)),
                nn.GroupNorm(output_dim // 8, output_dim, affine=True),
                nn.ConvTranspose3d(output_dim, output_dim, kernel_size=3, stride=maxstride, padding=1,
                                   output_padding=maxpadding)
            )
        else:
            if from_image:
                self.conv_block = nn.Sequential(
                    nn.Conv3d(input_dim, output_dim, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1)),
                    nn.GroupNorm(output_dim // 8, output_dim, affine=True),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(output_dim, output_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                    nn.GroupNorm(output_dim // 8, output_dim, affine=True),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(output_dim, output_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                )
            else:
                self.conv_block = nn.Sequential(
                    nn.GroupNorm(input_dim // 8, input_dim, affine=True),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(input_dim, output_dim, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1)),

                    nn.GroupNorm(output_dim // 8, output_dim, affine=True),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(output_dim, output_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),

                    nn.GroupNorm(output_dim // 8, output_dim, affine=True),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(output_dim, output_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                )

            self.conv_skip = nn.Sequential(
                nn.Conv3d(input_dim, output_dim, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1)),
                nn.GroupNorm(output_dim // 8, output_dim, affine=True),
            )

        #self.qam=QAM()

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


def _upsample_like(src,tar):
    src = F.upsample(src,size=tar.shape[2:] ,mode='trilinear', align_corners=False)
    return src

class SmallDARUnet(nn.Module):
    def __init__(self,num_classes=14):
        super(SmallDARUnet, self).__init__()
        channels=[16,32,64,128,256]

        self.L1_fromimg = ResBlock(1,channels[0],stride=1,from_image=True)
        self.L2_down = ResBlock(channels[0], channels[1], stride=2, anisotropy_stride=True)
        self.L3_down = ResBlock(channels[1], channels[2], stride=2, anisotropy_stride=True)
        self.L4_down = ResBlock(channels[2], channels[3], stride=2)
        self.L5_down = ResBlock(channels[3], channels[4], stride=2)
        self.L4_up = ResBlock( channels[4], channels[3],max=True)
        self.L3_up = ResBlock( channels[3], channels[2],max=True)
        self.L2_up = ResBlock( channels[2], channels[1], anisotropy_stride=True,max=True)
        self.L1_up = ResBlock( channels[1], channels[0], anisotropy_stride=True,max=True)
        self.sides=nn.ModuleList([nn.Conv3d(channels[0],num_classes,3,1,1),
                                  nn.Conv3d(channels[1], num_classes, 3, 1, 1),
                                  nn.Conv3d(channels[2], num_classes, 3, 1, 1),
                                  nn.Conv3d(channels[3], num_classes, 3, 1, 1),
                                  nn.Conv3d(channels[4], num_classes, 3, 1, 1),
        ])
        self.outconv = nn.Conv3d(num_classes*5, num_classes, 3, padding=1)
    def forward(self, x):

        #------------small--------------------
        x0_0 = self.L1_fromimg(x)
        x1_0 = self.L2_down(x0_0)
        x2_0 = self.L3_down(x1_0)
        x3_0 = self.L4_down(x2_0)
        x4_1 = self.L5_down(x3_0)

        x3_2 = self.L4_up(x4_1)
        x2_3 = self.L3_up(x3_2)
        x1_4 = self.L2_up(x2_3)
        x0_5 = self.L1_up(x1_4)

        d1 = self.sides[0](x0_5)
        d2 = self.sides[1](x1_4)
        d3 = self.sides[2](x2_3)
        d4 = self.sides[3](x3_2)
        d5 = self.sides[4](x4_1)

        d2 = _upsample_like(d2, d1)
        d3 = _upsample_like(d3, d1)
        d4 = _upsample_like(d4, d1)
        d5 = _upsample_like(d5, d1)
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5), 1))
        return d0, d1, d2, d3, d4, d5

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    model=SmallDARUnet().cuda()
    input=torch.rand((2,1,32,256,256)).cuda()
    print(model(input).shape)

