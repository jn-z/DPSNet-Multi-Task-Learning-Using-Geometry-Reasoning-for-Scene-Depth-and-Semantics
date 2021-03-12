import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#-----------------------------------------
# Encoder block down class - one block of conv, batchnorm and prelu, bringing the input image down to feature level, halving the size
class Encoder_block_down(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(Encoder_block_down, self).__init__()

        self.downConv = nn.Conv2d(input_channel, output_channel, kernel_size=4, stride=2, padding=1, bias=True)
        self.bn = torch.nn.BatchNorm2d(output_channel)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.prelu(self.bn(self.downConv(x)))
        return x
#-----------------------------------------

# Encoder block in class - one block of conv, halving the size of the image
class Encoder_block_in(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(Encoder_block_in, self).__init__()

        self.downConv = torch.nn.Conv2d(input_channel, output_channel, kernel_size=4, stride=2, padding=1, bias=True)

    def forward(self, x):
        x = self.downConv(x)
        return x
#-----------------------------------------

# Encoder block in class - one block of conv, halving the size of the image
class Encoder_block_bottle(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Encoder_block_bottle, self).__init__()
        self.bott_conv = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(output_channel)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.prelu(self.bn(self.bott_conv(x)))
        return x
#-----------------------------------------

# Full encoder class - encoding the image and returning feature sets at every level for future concatenation in the decoder
class Encoder(nn.Module):

    def __init__ (self, n_channels):
        super(Encoder, self).__init__()

        self.inc = Encoder_block_in(n_channels, 32) #[1,32,256,64]
        self.down1 = Encoder_block_down(32, 64)#[1,64,128,32]
        self.down2 = Encoder_block_down(64, 128)#[1,128,64,16]
        self.down3 = Encoder_block_down(128, 256)#[1,256,32,8]
        self.down4 = Encoder_block_down(256, 256)#[1,256,16,4]
        self.down5 = Encoder_block_down(256, 256)##[1,256,8,2]
        self.bneck = Encoder_block_bottle(256, 256)#[1,256,8,2]

    def forward(self, x):

        x1 = self.inc(x) #[1,32,256,64]
        x2 = self.down1(x1)#[1,64,128,32]
        x3 = self.down2(x2)#[1,128,64,16]
        x4 = self.down3(x3)#[1,256,32,8]
        x5 = self.down4(x4)#[1,256,16,4]
        x6 = self.down5(x5)#[1,256,8,2]
        x7 = self.bneck(x6)#[1,256,8,2]

        return [x1, x2, x3, x4, x5, x6, x7]
#-----------------------------------------

# Decoder block up class for the first decoder - one block of upsample, conv, batchnorm and prelu, bringing the feature produced by the encoderSSS up to bigger feature level, doubling the size
class Decoder_block_up_one(nn.Module):

    def __init__(self, input_channel_dec, input_channel_enc, output_channel_last):
        super(Decoder_block_up_one, self).__init__()

        output_channel_first = input_channel_dec + (input_channel_enc)
        self.conv_first = nn.Conv2d(input_channel_dec + (input_channel_enc), output_channel_first, kernel_size=3, stride=1, padding=1, bias=True)


        self.conv_last = nn.Conv2d(output_channel_first, output_channel_last, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_first = nn.BatchNorm2d(output_channel_first)
        self.bn_last = nn.BatchNorm2d(output_channel_last)
        self.prelu = nn.PReLU()

    def forward(self, x_dec, x_enc_1):

        x = torch.cat([x_dec, x_enc_1], dim=1)
        x = F.interpolate(x, scale_factor=2)
        x = self.prelu(self.bn_first(self.conv_first(x)))
        x = self.prelu(self.bn_last(self.conv_last(x))) #[1,256,16,4]

        return x
#-----------------------------------------

# Decoder block up class for the second decoder - one block of upsample, conv, batchnorm and prelu, bringing the feature produced by the encoderSSS up to bigger feature level, doubling the size
class Decoder_block_up_two(nn.Module):

    def __init__(self, input_channel_dec, input_channel_enc, output_channel_first, output_channel_last):
        super(Decoder_block_up_two, self).__init__()

        self.conv_first = nn.Conv2d(input_channel_dec + (input_channel_enc), output_channel_first, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_last = nn.Conv2d(output_channel_first, output_channel_last, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_first = nn.BatchNorm2d(output_channel_first)
        self.bn_last = nn.BatchNorm2d(output_channel_last)
        self.prelu = nn.PReLU()

    def forward(self, x_dec, x_enc_1):

        x = torch.cat([x_dec, x_enc_1], dim=1)
        x = F.interpolate(x, scale_factor=2)

        x = self.prelu(self.bn_first(self.conv_first(x)))
        x = self.prelu(self.bn_last(self.conv_last(x)))

        return x
#-----------------------------------------

# Decoder block bottleneck class - one block of conv, batchnorm and prelu, concatenating and conv-ing the feature produced by the encoder. No change to the size of the feautre other than the number of channels.
class Decoder_block_bottle(nn.Module):

    def __init__(self, input_channel, output_channel_pre, output_channel_post):
        super(Decoder_block_bottle, self).__init__()

        self.conv_pre = nn.Conv2d(input_channel, output_channel_pre, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_post = nn.Conv2d(output_channel_pre, output_channel_post, kernel_size=3, stride=1, padding=1, bias=True)

        self.bn_pre = nn.BatchNorm2d(output_channel_pre)
        self.bn_post = nn.BatchNorm2d(output_channel_post)
        self.prelu = nn.PReLU()

    def forward(self, x_enc_1):#, x_enc_2):

        x_enc_1 = self.prelu(self.bn_pre(self.conv_pre(x_enc_1)))
        #x_enc_2 = self.prelu(self.bn_pre(self.conv_pre(x_enc_2)))

        #x = torch.cat([x_enc_1, x_enc_2], dim=1)
        x = self.prelu(self.bn_post(self.conv_post(x_enc_1)))

        return x
#-----------------------------------------

# Decoder part one class - decoder the features coming from three separate encoders to produce image sized feature which will later be used for depth estimation and segmentation.
class Decoder_part_one(nn.Module):

    def __init__(self, n_feat=1024):
        super(Decoder_part_one, self).__init__()

        self.bneck = Decoder_block_bottle(256, 256, 256)

        self.up1 = Decoder_block_up_one(256, 256, 256)
        self.up2 = Decoder_block_up_one(256, 256, 256)
        self.up3 = Decoder_block_up_one(256, 256, 512)

        self.feat_out = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(512, n_feat, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(n_feat, eps=1e-5, momentum=0.1, affine=True),
            nn.PReLU(),
        )

    def forward(self, ix7s, ix6s, ix5s, ix4s):

        ox1 = self.bneck(ix7s) #[1,256,8,2]

        ox2 = self.up1(ox1, ix6s) #[1,256,16,4]
        ox3 = self.up2(ox2, ix5s) #[1,256,32,8]
        ox4 = self.up3(ox3, ix4s) #[1,512,64,16]
        ox5 = self.feat_out(ox4) #[1,1024,64,16]

        return ox5
#-----------------------------------------

# This block is used for the depth pyramid scales.
class Depth_Output_Block(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=3, use_bias=True):
        super(Depth_Output_Block, self).__init__()

        model = [
            nn.ReflectionPad2d(int(kernel_size/2)),
            nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, padding=0, bias=use_bias),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
#-----------------------------------------

# Full decoder part two class - starts from the end of the first part of the decoder and upsamples features right up to original image size. This has been separated from the first part to allow for independant learning for the two depth estimation and segmantation tasks. The last layer depends on whether the task at hand is monocular depth estimation or segmentation
class Decoder_part_two_instance(nn.Module):

    def __init__(self, n_feat=1024, num_ref_instance=None):
        super(Decoder_part_two_instance, self).__init__()

        self.inc_dec = Encoder_block_bottle(n_feat, 640)

        self.up4 = Decoder_block_up_two(640, 128, 512, 320)
        self.up5 = Decoder_block_up_two(320, 64, 256, 160)
        self.up6 = Decoder_block_up_two(160, 32, 128, 64)

        self.num_ref_instance = num_ref_instance

        if num_ref_instance is None:
            self.pre_sigi_doosh = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.sigi_doosh_doosh = nn.Sigmoid()

            self.output4 = nn.Conv2d(640, 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.output3 = nn.Conv2d(320, 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.output2 = nn.Conv2d(160, 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.pyramid_outs = []

        elif num_ref_instance is not None:
            self.pre_sigi_doosh = nn.Conv2d(64, num_ref_instance, kernel_size=3, stride=1, padding=1, bias=True)
            self.sigi_doosh_doosh = nn.Sigmoid()

            self.output4 = nn.Conv2d(640, num_ref_instance, kernel_size=3, stride=1, padding=1, bias=True)
            self.output3 = nn.Conv2d(320, num_ref_instance, kernel_size=3, stride=1, padding=1, bias=True)
            self.output2 = nn.Conv2d(160, num_ref_instance, kernel_size=3, stride=1, padding=1, bias=True)
            self.pyramid_outs = []

    def forward(self, ox5, ix3s, ix2s, ix1s):
        #ox5 = torch.cat([ox5_,_ox5],1)
        ox6 = self.inc_dec(ox5) #[640, 64, 16]
        ox7 = self.up4(ox6, ix3s)  # [1,320,128,32]
        ox8 = self.up5(ox7, ix2s)  # [1,160,256,64]
        ox9 = self.up6(ox8, ix1s)  # [1,64,512,128]

        pre_output_1 = self.pre_sigi_doosh(ox9)  # [1,num_classes,512,128]
        pre_output_2 = self.output2(ox8)  # [1,num_classes,256,64]
        pre_output_3 = self.output3(ox7)  # [1,num_classes,128,32]
        pre_output_4 = self.output4(ox6)  # [1,num_classes,64,16]

        output_1 = self.sigi_doosh_doosh(pre_output_1)  # [1,num_classes,512,128]
        output_2 = self.sigi_doosh_doosh(pre_output_2)  # [1,num_classes,256,64]
        output_3 = self.sigi_doosh_doosh(pre_output_3)  # [1,num_classes,128,32]
        output_4 = self.sigi_doosh_doosh(pre_output_4)  # [1,num_classes,64,16]

        return [output_1, output_2, output_3, output_4]

# Full decoder part two class - starts from the end of the first part of the decoder and upsamples features right up to original image size. This has been separated from the first part to allow for independant learning for the two depth estimation and segmantation tasks. The last layer depends on whether the task at hand is monocular depth estimation or segmentation
class Decoder_part_two(nn.Module):

    def __init__(self, n_feat=1024, num_classes=None):
        super(Decoder_part_two, self).__init__()

        self.inc_dec = Encoder_block_bottle(n_feat, 640) #[1,640,64,16]

        self.up4 = Decoder_block_up_two(640, 128, 512, 320)
        self.up5 = Decoder_block_up_two(320, 64, 256, 160)
        self.up6 = Decoder_block_up_two(160, 32, 128, 64)

        self.num_classes = num_classes

        if num_classes is None:
            self.pre_sigi_doosh = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.sigi_doosh_doosh = nn.Sigmoid()

            self.output4 = nn.Conv2d(640, 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.output3 = nn.Conv2d(320, 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.output2 = nn.Conv2d(160, 1, kernel_size=3, stride=1, padding=1, bias=True)
            self.pyramid_outs = []

        elif num_classes is not None:
            self.pre_sigi_doosh = nn.Conv2d(64,  num_classes, kernel_size=3, stride=1, padding=1, bias=True)
            self.sigi_doosh_doosh = nn.Sigmoid()

            self.output4 = nn.Conv2d(640,  num_classes, kernel_size=3, stride=1, padding=1, bias=True)
            self.output3 = nn.Conv2d(320,  num_classes, kernel_size=3, stride=1, padding=1, bias=True)
            self.output2 = nn.Conv2d(160,  num_classes, kernel_size=3, stride=1, padding=1, bias=True)
            self.pyramid_outs = []

    def forward(self, ox5, ix3s, ix2s, ix1s):
        #ox5 [1,1024,64,16]
        ox6 = self.inc_dec(ox5) #[1,640,64,16]
        ox7 = self.up4(ox6, ix3s) #[1,320,128,32]
        ox8 = self.up5(ox7, ix2s) #[1,160,256,64]
        ox9 = self.up6(ox8, ix1s) #[1,64,512,128]

        pre_output_1 = self.pre_sigi_doosh(ox9) #[1,num_classes,512,128]
        pre_output_2 = self.output2(ox8) #[1,num_classes,256,64]
        pre_output_3 = self.output3(ox7) #[1,num_classes,128,32]
        pre_output_4 = self.output4(ox6) #[1,num_classes,64,16]

        output_1 = self.sigi_doosh_doosh(pre_output_1) #[1,num_classes,512,128]
        output_2 = self.sigi_doosh_doosh(pre_output_2) #[1,num_classes,256,64]
        output_3 = self.sigi_doosh_doosh(pre_output_3) #[1,num_classes,128,32]
        output_4 = self.sigi_doosh_doosh(pre_output_4) #[1,num_classes,64,16]

        return [output_1, output_2, output_3, output_4]
#-----------------------------------------

# Defines the discriminator with the specified arguments based on the code in https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.
class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
#-----------------------------------------

# This class is taken from the pyTorch implementation of spynet in https://github.com/sniklaus/pytorch-spynet
#-----------------------------------------
class FlowNetwork(torch.nn.Module):
    def __init__(self, arguments_strModel):
        super(FlowNetwork, self).__init__()

        class Preprocess(torch.nn.Module):
            def __init__(self):
                super(Preprocess, self).__init__()

            def forward(self, tensorInput):
                tensorBlue = tensorInput[:, 0:1, :, :] - 0.406
                tensorGreen = tensorInput[:, 1:2, :, :] - 0.456
                tensorRed = tensorInput[:, 2:3, :, :] - 0.485

                tensorBlue = tensorBlue / 0.225
                tensorGreen = tensorGreen / 0.224
                tensorRed = tensorRed / 0.229

                return torch.cat([tensorRed, tensorGreen, tensorBlue], 1)

        class Basic(torch.nn.Module):
            def __init__(self, intLevel):
                super(Basic, self).__init__()

                self.moduleBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )

            def forward(self, tensorInput):
                return self.moduleBasic(tensorInput)

        class Backward(torch.nn.Module):
            def __init__(self):
                super(Backward, self).__init__()

            def forward(self, tensorInput, tensorFlow):
                if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != tensorFlow.size(0) or self.tensorGrid.size(2) != tensorFlow.size(2) or self.tensorGrid.size(3) != tensorFlow.size(3):
                    tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
                    tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

                    self.tensorGrid = torch.cat([tensorHorizontal, tensorVertical], 1).cuda()

                tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

                return torch.nn.functional.grid_sample(input=tensorInput, grid=(self.tensorGrid + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')

        self.modulePreprocess = Preprocess()

        self.moduleBasic = torch.nn.ModuleList([Basic(intLevel) for intLevel in range(6)])

        self.moduleBackward = Backward()

    def forward(self, tensorFirst, tensorSecond):
        tensorFlow = []

        tensorFirst = [self.modulePreprocess(tensorFirst)]
        tensorSecond = [self.modulePreprocess(tensorSecond)]

        for intLevel in range(5):
            if tensorFirst[0].size(2) > 32 or tensorFirst[0].size(3) > 32:
                tensorFirst.insert(0, torch.nn.functional.avg_pool2d(input=tensorFirst[0], kernel_size=2, stride=2))
                tensorSecond.insert(0, torch.nn.functional.avg_pool2d(input=tensorSecond[0], kernel_size=2, stride=2))

        tensorFlow = torch.FloatTensor(tensorFirst[0].size(0), 2, int(math.floor(tensorFirst[0].size(2) / 2.0)), int(math.floor(tensorFirst[0].size(3) / 2.0))).zero_().cuda()

        for intLevel in range(len(tensorFirst)):

            tensorUpsampled = torch.nn.functional.interpolate(input=tensorFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if tensorUpsampled.size(2) != tensorFirst[intLevel].size(2): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
            if tensorUpsampled.size(3) != tensorFirst[intLevel].size(3): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

            tensorFlow = self.moduleBasic[intLevel](torch.cat([ tensorFirst[intLevel], self.moduleBackward(tensorSecond[intLevel], tensorUpsampled), tensorUpsampled], 1)) + tensorUpsampled


        return tensorFlow
#-----------------------------------------

# Returns the adversarial loss based on the code in https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.
class AdvLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):

        super(AdvLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError('The adversarial loss you want (%s) has not yet been implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):

        if target_is_real:
            target_tensor = self.real_label

        else:
            target_tensor = self.fake_label

        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):

        if self.gan_mode in ['lsgan', 'vanilla']:

            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)

        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()

        return loss
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
