"""
    Implementation of the Laplacian regularizer.

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

import torch


class LaplacianRegularizer(torch.nn.Module):
    """
        Laplacian-regularized Loss across joints or features.
        W is built from the stand-alone adjacency matrix of NTU dataset.
        The regularizer Rskel needs to be weighted with an additive penalty term
        for the loss.
    """

    def __init__(self, penalty):
        super(LaplacianRegularizer, self).__init__()
        self.penalty = penalty

        # NTU 60 and NTU120 connectivity pairs jor joints tuples
        ntu_skeleton_bone_pairs = tuple((i, j)
                                        for (i, j) in ((0, 1), (0, 12),
                                                       (0, 16), (1, 20),
                                                       (2, 3), (2, 20),
                                                       (4, 5), (4, 20),
                                                       (5, 6), (6, 7),
                                                       (7, 21), (7, 22),
                                                       (8, 9), (8, 20),
                                                       (9, 10), (10, 11),
                                                       (11, 23), (11, 24),
                                                       (12, 13), (13, 14),
                                                       (14, 15), (16, 17),
                                                       (17, 18), (18, 19)))

        # Initialize a temporary adjacency matrix with zeros
        W_temp = torch.zeros((25, 25)).int()

        # Fill it with ones where joints are connected together
        for i, j in ntu_skeleton_bone_pairs:
            W_temp[i][j] = 1

        # Get it symmetrical
        W_temp = W_temp + W_temp.T

        # Adjacency matrix W to be used for Laplacian regularizer
        self.W = torch.tensor(W_temp, dtype=torch.float,
                              requires_grad=False, device='cuda')

    def forward(self, target):
        """ Laplacian regularization across joints """
        Rskel = 0

        L = torch.diag(torch.sum(self.W, dim=1)) - self.W

        for sample in target.permute(0, 1, 3, 2):
            for channel in sample:
                channel = torch.mean(channel, dim=0)
                Rskel += channel.matmul(L).matmul(channel)

        return Rskel * self.penalty
