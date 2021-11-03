"""
    Implementation of the Self-Supervised Viewpoint Invariance function.

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


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class RotationZ(torch.nn.Module):
    def __init__(self):
        super(RotationZ, self).__init__()
        self._2pi = torch.Tensor([2 * math.pi])

    def sample_rotate(self, x):
        """ Sample rotation: e ~ N(0, 2pi) """
        alpha_rot = torch.rand(1) * self._2pi
        Z_rot = torch.tensor([[math.cos(alpha_rot), -math.sin(alpha_rot), 0],
                              [math.sin(alpha_rot), math.cos(alpha_rot), 0],
                              [0, 0, 1]], requires_grad=False)
        if x.is_cuda:
            Z_rot = Z_rot.cuda()
            alpha_rot = alpha_rot.cuda()
        x = x.permute(2, 1, 0)
        for i in range(x.shape[0]):
            x[i] = x[i].matmul(Z_rot)
        x = x.permute((2, 1, 0))
        return x, alpha_rot

    def forward(self, x):
        alpha_rot = torch.empty(0)
        if x.is_cuda:
            alpha_rot = alpha_rot.cuda()
        for sample in range(x.shape[0]):
            x[sample], sample_alpha_rot = self.sample_rotate(x[sample])
            alpha_rot = torch.cat((alpha_rot, sample_alpha_rot), dim=0)
        return x, alpha_rot
