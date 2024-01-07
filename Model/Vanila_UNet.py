import torch
import torch.nn as nn


class VanilaUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(VanilaUNet, self).__init__()
        '''
        Vanila U-Net model
            - in_channels = Chanel of Image
            - out_channels = Number of class
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.enc1 = self.conv_block(self.in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.pool = nn.MaxPool2d(2, 2)

        self.bottel_neck = self.conv_block(512, 1024)

        self.upconv1 = self.upconv_block(128, 64)
        self.upconv2 = self.upconv_block(256, 128)
        self.upconv3 = self.upconv_block(512, 256)
        self.upconv4 = self.upconv_block(1024, 512)

        self.dec4 = self.conv_block(1024,512)
        self.dec3 = self.conv_block(512,256)
        self.dec2 = self.conv_block(256, 128)
        self.dec1 = self.conv_block(128, 64)

        self.head = nn.Conv2d(64, self.out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU()
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU()
        )
    
    def copy_and_crop(self, input_feature, target_feature):
        
        height, width = input_feature.shape[2:]
        target_height, target_width = target_feature.shape[2:]
        
        start_h = int((height - target_height) / 2)
        start_w = int((width - target_width) / 2)

        cropped_map = input_feature[:, :, start_h:start_h+target_height, start_w : start_w+target_width]

        return cropped_map
    
    def forward(self, x):
        
        # 1st down-sampling
        conv1 = self.enc1(x)
        down1 = self.pool(conv1)

        # 2nd down-sampling
        conv2 = self.enc2(down1)
        down2 = self.pool(conv2)

        # 3rd down-sampling
        conv3 = self.enc3(down2)
        down3 = self.pool(conv3)

        # 4th down-sampling
        conv4 = self.enc4(down3)
        down4 = self.pool(conv4)

        # Bottle-neck
        bottle = self.bottel_neck(down4)

        # 4th up-sampling
        up4 = self.upconv4(bottle)
        concat4 = torch.concat([self.copy_and_crop(conv4, up4), up4], 1)
        deconv4 = self.dec4(concat4)

        # 3rd up-sampling
        up3 = self.upconv3(deconv4)
        concat3 = torch.concat([self.copy_and_crop(conv3, up3), up3], 1)
        deconv3 = self.dec3(concat3)

        # 2nd up-sampling
        up2 = self.upconv2(deconv3)
        concat2 = torch.concat([self.copy_and_crop(conv2, up2), up2], 1)
        deconv2 = self.dec2(concat2)

        # 1st up-sampling
        up1 = self.upconv1(deconv2)
        concat1 = torch.concat([self.copy_and_crop(conv1, up1), up1], 1)
        deconv1 = self.dec1(concat1)

        # out
        seg_map = self.head(deconv1)
        out = self.copy_and_crop(seg_map, torch.randn(1, 2, 256, 256))
        return out


    
