import torch


class Norm(torch.nn.Module):
    def __init__(self, num_features, norm=None):
        super(Norm, self).__init__()
        if norm == 'batch':
            self.norm = torch.nn.BatchNorm2d(num_features=num_features)
        elif norm == 'instance':
            self.norm = torch.nn.InstanceNorm2d(num_features=num_features)
        else:
            self.norm = None

    def forward(self, x):
        if self.norm is None:
            return x
        else:
            return self.norm(x)


class Activation(torch.nn.Module):
    def __init__(self, activation=None):
        super(Activation, self).__init__()
        if activation == 'relu':
            self.act = torch.nn.ReLU(inplace=True)
        elif activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        if self.act is None:
            return x
        else:
            return self.act(x)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, activation=None, norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=bias)
        self.norm = Norm(num_features=out_channels, norm=norm)
        self.activation = Activation(activation=activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, activation=None, norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding=padding,
                                               bias=bias)
        self.norm = Norm(num_features=out_channels, norm=norm)
        self.activation = Activation(activation=activation)

    def forward(self, x):
        out = self.deconv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_channels, kernel_size, stride, padding, bias=True, activation=None, norm=None):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=num_channels,
                                     out_channels=num_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     bias=bias)
        self.conv2 = torch.nn.Conv2d(in_channels=num_channels,
                                     out_channels=num_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     bias=bias)
        self.norm = Norm(num_features=num_channels, norm=norm)
        self.activation = Activation(activation=activation)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm(out)
        out = torch.add(out, residual)
        out = self.activation(out)

        return out


class UpBlock(torch.nn.Module):
    def __init__(self, num_channels, kernel_size, stride, padding, bias=True, activation=None, norm=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(in_channels=num_channels,
                                    out_channels=num_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=bias,
                                    activation=activation,
                                    norm=norm)
        self.up_conv2 = ConvBlock(in_channels=num_channels,
                                  out_channels=num_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  bias=bias,
                                  activation=activation,
                                  norm=norm)
        self.up_conv3 = DeconvBlock(in_channels=num_channels,
                                    out_channels=num_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=bias,
                                    activation=activation,
                                    norm=norm)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(torch.nn.Module):
    def __init__(self, num_channels, kernel_size, stride, padding, bias=True, activation=None, norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(in_channels=num_channels,
                                    out_channels=num_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=bias,
                                    activation=activation,
                                    norm=norm)
        self.down_conv2 = DeconvBlock(in_channels=num_channels,
                                      out_channels=num_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      bias=bias,
                                      activation=activation,
                                      norm=norm)
        self.down_conv3 = ConvBlock(in_channels=num_channels,
                                    out_channels=num_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=bias,
                                    activation=activation,
                                    norm=norm)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0
