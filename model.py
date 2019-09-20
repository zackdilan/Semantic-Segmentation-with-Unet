import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):


    def contract_block(self,in_ch,out_ch):


        block = nn.Sequential(
                nn.Conv2d(in_ch,out_ch,kernel_size=3,padding = 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch,out_ch,kernel_size=3,padding = 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch)
                )
        return block

    def expansive_block(self,in_ch ,mid_ch ,out_ch):


        block = nn.Sequential(
                nn.Conv2d(in_ch,mid_ch,kernel_size=3,padding = 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(mid_ch),
                nn.Conv2d(mid_ch,mid_ch,kernel_size=3,padding = 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(mid_ch),
                nn.ConvTranspose2d(mid_ch,out_ch,kernel_size = 3,stride = 2,padding = 1,output_padding = 1)
                )

        return block

    def final_block(self, in_ch, mid_ch , out_ch):

        block = nn.Sequential(
                nn.Conv2d(in_ch,mid_ch,kernel_size=3,padding = 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(mid_ch),
                nn.Conv2d(mid_ch,mid_ch,kernel_size=3,padding = 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(mid_ch),
                nn.Conv2d(mid_ch,out_ch,kernel_size=3,padding=1)
                )
        return block



    def __init__(self,in_channel,no_classes):
        super (Unet,self).__init__()
        # encode part
        self.conv_encode1 = self.contract_block(in_ch = in_channel,out_ch = 64)
        self.downsample1  = nn.MaxPool2d(kernel_size = 2)
        self.conv_encode2 = self.contract_block(in_ch = 64 ,out_ch = 128)
        self.downsample2  = nn.MaxPool2d(kernel_size = 2)
        self.conv_encode3 = self.contract_block(in_ch = 128 ,out_ch = 256)
        self.downsample3  = nn.MaxPool2d(kernel_size = 2)
        self.conv_encode4 = self.contract_block(in_ch = 256 ,out_ch = 512)
        self.downsample4  = nn.MaxPool2d(kernel_size = 2)


        # bottle neck part
        self.bottleneck = nn.Sequential(
                          nn.Conv2d(512,1024,kernel_size=3,padding= 1),
                          nn.ReLU(inplace=True),
                          nn.BatchNorm2d(1024),
                          nn.Conv2d(1024,1024,kernel_size=3,padding = 1),
                          nn.ReLU(inplace=True),
                          nn.BatchNorm2d(1024),
                          nn.ConvTranspose2d(1024,512,kernel_size = 3,stride = 2,padding = 1,output_padding = 1)
                            )

        # decode part
        self.conv_decode4 = self.expansive_block(in_ch = 1024,mid_ch = 512 ,out_ch = 256)
        self.conv_decode3 = self.expansive_block(in_ch = 512,mid_ch = 256 ,out_ch = 128)
        self.conv_decode2 = self.expansive_block(in_ch = 256,mid_ch = 128 ,out_ch = 64)
        self.final_layer  = self.final_block(in_ch = 128,mid_ch = 64 ,out_ch = no_classes)


    def crop_and_concat(self,upsampled,bypass,crop = False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass,(-c,-c,-c,-c))

        else:
            return torch.cat((upsampled,bypass),1) # concatenate along column dimension

    def forward(self,x):
            # encode

            encode_layer1 = self.conv_encode1(x)
            encode_pool1  = self.downsample1(encode_layer1)

            encode_layer2 = self.conv_encode2(encode_pool1 )
            encode_pool2  = self.downsample2(encode_layer2)

            encode_layer3 = self.conv_encode3(encode_pool2 )
            encode_pool3  = self.downsample3(encode_layer3)

            encode_layer4 = self.conv_encode4(encode_pool3 )
            encode_pool4  = self.downsample3(encode_layer4)

            # Bottleneck

            bottleneck = self.bottleneck(encode_pool4)

            ##print("bottleneck size {}".format(bottleneck.shape),"encode_layer3 {}".format(encode_layer3.shape))

            # Decode
            skip_connection1 = self.crop_and_concat(bottleneck,encode_layer4,crop=False)
            decode_layer4    = self.conv_decode4(skip_connection1)

            skip_connection2 = self.crop_and_concat(decode_layer4,encode_layer3, crop=False)
            decode_layer3    = self.conv_decode3(skip_connection2)

            skip_connection3 = self.crop_and_concat(decode_layer3,encode_layer2,crop=False)
            decode_layer2    = self.conv_decode2(skip_connection3)

            skip_connection4 = self.crop_and_concat(decode_layer2,encode_layer1,crop =False)
            final_layer      = self.final_layer(skip_connection4)


            return final_layer
