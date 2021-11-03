"""
    Miscellaneous functions for the main script.

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

import numpy
import torch
import argparse


def argument_parser():
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=0)

    # Model settings
    parser.add_argument('--layers', type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument('--ls', type=int, default=2048)
    parser.add_argument('--SSVI-penalty', type=float, default=1e-3)

    # Path and account settings
    parser.add_argument('--data-path', type=str, default='./dataset')
    parser.add_argument('--wandb-user', type=str, default='your_wandb_username')

    # Ablation settings
    parser.add_argument('--split', type=str, default='xsub60',
                        choices=['xsub60', 'xview60', 'xsub120', 'xset120'])
    parser.add_argument('--method', type=str, default='AE_L',
                        choices=['AE', 'AE_L', 'GRAE_L'])

    args = parser.parse_args()
    return args


def log(s, nl=True):
    print(s, end='\n' if nl else '')


def random_seed(seed):
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += val.detach().cpu()
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


class NTUDatasetList(torch.utils.data.Dataset):
    def __init__(self, tensors):
        assert all(len(tensors[0]) == len(tensor) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        return x, y

    def __len__(self):
        return len(self.tensors[0])
