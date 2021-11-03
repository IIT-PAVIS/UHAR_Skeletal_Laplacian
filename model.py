"""
    Implementation of the GRAE_L model.

    Giancarlo Paoletti
    Copyright 2021 Giancarlo Paoletti (giancarlo.paoletti@iit.it)
    Please, email me if you have any question.

    Disclaimer:
    The software is provided "as is", without warranty of any kind, express or
    implied, including but not limited to the warranties of merchantability,
    fitness for a particular purpose and noninfringement.
    In no event shall the authors, PAVIS or IIT be liable for any claim, damages
    or other liability, whether in an action of contract, tort or otherwise,
    arising from, out of or in connection with the software or the use or other
    dealings in the software.

    LICENSE:
    This project is licensed under the terms of the MIT license.
    This project incorporates material from the projects listed below
    (collectively, "Third Party Code").
    This Third Party Code is licensed to you under their original license terms.
    We reserves all other rights not expressly granted, whether by implication,
    estoppel or otherwise.

    Copyright (c) 2021 Giancarlo Paoletti, Jacopo Cavazza, Cigdem Beyan and
    Alessio Del Bue

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
    USE OR OTHER DEALINGS IN THE SOFTWARE.

    References
    [1] Giancarlo Paoletti, Jacopo Cavazza, Cigdem Beyan and Alessio Del Bue (2021).
        Unsupervised Human Action Recognition with Skeletal Graph Laplacian and Self-Supervised Viewpoints Invariance
        British Machine Vision Conference (BMVC).
"""

import math
import torch
import torch.nn as nn

from SSVI import GRL


class GRAE_L(nn.Module):
    def __init__(self, args, init=True, bias=True):
        super(GRAE_L, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.layer = args.layers
        self.SSVI = args.SSVI
        self.SSVI_penalty = args.SSVI_penalty
        self.init = init
        self.bias = bias

        # Initialize conv layers for the encoder
        self.conv1_1 = nn.Conv2d(3, self.layer[0],
                                 kernel_size=(3, 1), stride=1,
                                 bias=bias, padding=(1, 0))
        self.conv1_2 = nn.Conv2d(self.layer[0], self.layer[0],
                                 kernel_size=(1, 3), stride=1,
                                 bias=bias, padding=(0, 1))
        self.conv1_3 = nn.Conv2d(self.layer[0], self.layer[0],
                                 kernel_size=(3, 1), stride=1,
                                 bias=bias, padding=(1, 0))
        self.conv2_1 = nn.Conv2d(self.layer[0], self.layer[1],
                                 kernel_size=(3, 1), stride=1,
                                 bias=bias, padding=(1, 0))
        self.conv2_2 = nn.Conv2d(self.layer[1], self.layer[1],
                                 kernel_size=(1, 3), stride=1,
                                 bias=bias, padding=(0, 1))
        self.conv2_3 = nn.Conv2d(self.layer[1], self.layer[1],
                                 kernel_size=(3, 1), stride=1,
                                 bias=bias, padding=(1, 0))
        self.conv3_1 = nn.Conv2d(self.layer[1], self.layer[2],
                                 kernel_size=(3, 1), stride=1,
                                 bias=bias, padding=(1, 0))
        self.conv3_2 = nn.Conv2d(self.layer[2], self.layer[2],
                                 kernel_size=(1, 3), stride=1,
                                 bias=bias, padding=(0, 1))
        self.conv3_3 = nn.Conv2d(self.layer[2], self.layer[2],
                                 kernel_size=(3, 1), stride=1,
                                 bias=bias, padding=(1, 0))

        # Initialize deconv layers for the decoder
        self.deconv3_1 = nn.ConvTranspose2d(self.layer[2], self.layer[2],
                                            kernel_size=(3, 1), stride=1,
                                            bias=bias, padding=(1, 0))
        self.deconv3_2 = nn.ConvTranspose2d(self.layer[2], self.layer[2],
                                            kernel_size=(1, 3), stride=1,
                                            bias=bias, padding=(0, 1))
        self.deconv3_3 = nn.ConvTranspose2d(self.layer[2], self.layer[1],
                                            kernel_size=(3, 1), stride=1,
                                            bias=bias, padding=(1, 0))
        self.deconv2_1 = nn.ConvTranspose2d(self.layer[1], self.layer[1],
                                            kernel_size=(3, 1), stride=1,
                                            bias=bias, padding=(1, 0))
        self.deconv2_2 = nn.ConvTranspose2d(self.layer[1], self.layer[1],
                                            kernel_size=(1, 3), stride=1,
                                            bias=bias, padding=(0, 1))
        self.deconv2_3 = nn.ConvTranspose2d(self.layer[1], self.layer[0],
                                            kernel_size=(3, 1), stride=1,
                                            bias=bias, padding=(1, 0))
        self.deconv1_1 = nn.ConvTranspose2d(self.layer[0], self.layer[0],
                                            kernel_size=(3, 1), stride=1,
                                            bias=bias, padding=(1, 0))
        self.deconv1_2 = nn.ConvTranspose2d(self.layer[0], self.layer[0],
                                            kernel_size=(1, 3), stride=1,
                                            bias=bias, padding=(0, 1))
        self.deconv1_3 = nn.ConvTranspose2d(self.layer[0], 3,
                                            kernel_size=(3, 1), stride=1,
                                            bias=bias, padding=(1, 0))

        # Initialize latent space layer
        self.enc_fc = nn.Linear(self.layer[2] * 36, args.ls, bias=self.bias)
        self.dec_fc = nn.Linear(args.ls, self.layer[2] * 36, bias=self.bias)

        # Initialize regressor head for SSVI
        if self.SSVI:
            self.regressor_head = nn.Linear(args.ls, 1, bias=self.bias)

        # Initialize batchnorm layers for decoder
        self.bn0 = nn.BatchNorm2d(3)
        self.bn1 = nn.BatchNorm2d(self.layer[0])
        self.bn2 = nn.BatchNorm2d(self.layer[1])
        self.bn3 = nn.BatchNorm2d(self.layer[2])

        # Initialize weights if requested
        if self.init:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encoder(self, x):

        # Residual block #1
        identity = self.conv1_1(x)
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.relu(self.conv1_3(x))
        x += identity
        x = self.relu(x)
        x, self.i1 = self.pool(x)

        # Residual block #2
        identity = self.conv2_1(x)
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.relu(self.conv2_3(x))
        x += identity
        x = self.relu(x)
        x, self.i2 = self.pool(x)

        # Residual block #3
        identity = self.conv3_1(x)
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x += identity
        x = self.relu(x)
        x, self.i3 = self.pool(x)

        # Flatten and FC features
        x = self.enc_fc(torch.flatten(x, 1))

        return x

    def decoder(self, x):
        # Restore and reshape features for decoder
        x = self.dec_fc(x)
        x = self.bn3(x.view([-1, self.layer[2], 3, 12]))

        # Residual block #1
        self.unpool3 = torch.nn.MaxUnpool2d((2, 1), 2, (0, -1))
        x = self.unpool3(x, self.i3)
        identity = self.deconv3_3(x)
        x = self.relu(self.deconv3_1(x))
        x = self.relu(self.deconv3_2(x))
        x = self.relu(self.deconv3_3(x))
        x += identity
        x = self.bn2(self.relu(x))

        # Residual block #2
        self.unpool2 = torch.nn.MaxUnpool2d(2, 2)
        x = self.unpool2(x, self.i2)
        identity = self.deconv2_3(x)
        x = self.relu(self.deconv2_1(x))
        x = self.relu(self.deconv2_2(x))
        x = self.relu(self.deconv2_3(x))
        x += identity
        x = self.bn1(self.relu(x))

        # Residual block #3
        self.unpool1 = torch.nn.MaxUnpool2d((1, 2), 2, (-1, 0))
        x = self.unpool1(x, self.i1)
        identity = self.deconv1_3(x)
        x = self.relu(self.deconv1_1(x))
        x = self.relu(self.deconv1_2(x))
        x = self.relu(self.deconv1_3(x))
        x += identity
        x = self.bn0(x)

        return x

    def grl_mlp(self, x):
        return self.sigmoid(self.regressor_head(x)) * (2 * math.pi)

    def forward(self, x):
        ls = self.encoder(x)
        x_hat = self.decoder(ls)
        if self.SSVI:
            reverse_feature = GRL.apply(ls, self.SSVI_penalty)
            alpha_hat = self.grl_mlp(reverse_feature)
            return x_hat, ls, alpha_hat
        else:
            return x_hat, ls
