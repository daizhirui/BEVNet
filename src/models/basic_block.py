import torch.nn.modules as nn


class BasicConv2d(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
        relu=nn.LeakyReLU, batch_norm=True, dropout=False, dropout_p=0.5
    ):
        super(BasicConv2d, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(relu())
        if dropout:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BasicDeconv2d(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=2, padding=1,
        output_padding=1, relu=nn.LeakyReLU, batch_norm=True, dropout=False,
        dropout_p=0.5
    ):
        super(BasicDeconv2d, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                               padding, output_padding)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(relu())
        if dropout:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
