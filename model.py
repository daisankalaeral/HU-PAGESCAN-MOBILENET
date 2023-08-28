import torch
import numpy as np
import torch.nn.functional as F
from torchvision.utils import save_image
import lightning as pl
import cv2 as cv

class HU_PageScan(pl.LightningModule):
    def __init__(self, encoder_bloc_n = 5, decoder_bloc_n = 4):
        super().__init__()
        self.test_image_id = 0
        
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.encoder = torch.nn.ModuleList([encoder_block(2**(i-1)*32, 2**i*32) if i > 0 else encoder_block(1,32) for i in range(encoder_bloc_n-1)])
        self.middle = encoder_block(2**(encoder_bloc_n-2)*32, 2**(encoder_bloc_n-1)*32)

        self.decoder = torch.nn.ModuleList([decoder_block(2**i*32, 2**(i-1)*32) if i < decoder_bloc_n else decoder_block(2**i*32+2**(i-1)*32, 2**(i-1)*32) for i in range(decoder_bloc_n, 0,-1)])

        self.up_conv = torch.nn.ModuleList([torch.nn.ConvTranspose2d(2**(encoder_bloc_n-1)*32, 2**(encoder_bloc_n-1)*32, kernel_size=2, stride=2, padding=0, output_padding=0, groups=2)])

        channels_stuff = []
        for decoder_layer in self.decoder:
            input_channels = decoder_layer.layer[0].in_channels
            output_channels = decoder_layer.layer[0].out_channels
            channels_stuff.append((input_channels, output_channels))

        for stuff in channels_stuff[1:]:
            self.up_conv.append(torch.nn.ConvTranspose2d(stuff[0], stuff[1], kernel_size=2, stride=2, padding=0, output_padding=0, groups=2))
        
        self.final_conv = torch.nn.Conv2d(channels_stuff[-1][1], 1, kernel_size=1, stride=1)
        
        self.sigmoid = torch.nn.Sigmoid()
    
    def _common_step(self, image):
        # print(image.shape)
        x = image
        temp = False
        residual = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            saved_x = x
            x = self.max_pool(x)
            # print(x.shape)
            residual.append(saved_x)
        
        x = self.middle(x)
        for i, decoder_layer in enumerate(self.decoder):
            x = self.up_conv[i](x)
            x = torch.concat((x, residual.pop()), 1)
            x = decoder_layer(x)
            # print(x.shape)

        output = self.final_conv(x)
        output = self.sigmoid(output)
        # print(output.shape)

        return output

    def forward(self,  image):
        return self._common_step(image)

    def training_step(self, batch, batch_idx):
        image, mask = batch

        y_pred = self._common_step(image)

        loss = dice_coef_loss(y_pred, mask)

        self.log_dict(
            {
                "train_loss": loss
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss
    
    def validation_step(self, batch, batch_idx):
        image, mask = batch

        y_pred = self._common_step(image)

        loss = dice_coef_loss(y_pred, mask)

        self.log_dict(
            {
                "valid_loss": loss
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss
    
    def test_step(self, batch, batch_idx):
        image, mask = batch

        y_pred = self._common_step(image)

        loss = dice_coef_loss(y_pred, mask)

        self.log_dict(
            {
                "test_loss": loss
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        
        haha = y_pred
        haha = haha.cpu()
        haha = haha.type(torch.float32)
        # haha = cv.cvtColor(haha, cv.COLOR_BGR2GRAY)
        save_image(mask.cpu(), f"/notebooks/{self.test_image_id}_gd.png")
        save_image(haha, f"/notebooks/{self.test_image_id}_hat.png")
        self.test_image_id += 1

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-4)
        return [optimizer]

    def make_up_conv_layers(self):
        for decoder_layer in self.decoder:
            print(decoder_layer.layer[0].in_channels)

class encoder_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias = True),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias = True),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )

    def forward(self, input):
        output = self.layer(input)
        return output

class decoder_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=2, bias = True),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=2, bias = True),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )

    def forward(self, input):
        output = self.layer(input)
        return output

def dice_coef(y_true, y_pred, smooth=1000.0):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)