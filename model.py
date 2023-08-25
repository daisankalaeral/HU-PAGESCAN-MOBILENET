import torch
import numpy as np

class HU_PageScan(torch.nn.Module):
    def __init__(self, encoder_bloc_n = 5, decoder_bloc_n = 4):
        super().__init__()
        
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
        
        self.final_conv = torch.nn.Conv2d(channels_stuff[-1][1], 1, kernel_size=1, stride=1)

        for stuff in channels_stuff[1:]:
            self.up_conv.append(torch.nn.ConvTranspose2d(stuff[0], stuff[1], kernel_size=2, stride=2, padding=0, output_padding=0, groups=2))
        
    def forward(self,  input):
        x = input
        temp = False
        residual = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            saved_x = x
            x = self.max_pool(x)
            residual.append(saved_x)
        
        x = self.middle(x)
        for i, decoder_layer in enumerate(self.decoder):
            x = self.up_conv[i](x)
            x = torch.concat((x, residual.pop()), 1)
            x = decoder_layer(x)

        x = self.final_conv(x)

        return x

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

model = HU_PageScan(5,4)
torch.save(model.state_dict(), "test.pth")
# x = torch.rand((1,1,512,512))
# print(model(x))

# model.eval()
# model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])
# print(params)