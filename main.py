"""
    Main script.

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
import wandb
import warnings

from time import time
from sklearn.neighbors import KNeighborsClassifier

from exec import train, test
from model import GRAE_L
from Rskel import LaplacianRegularizer
from SSVI import RotationZ
from utils import argument_parser, log, random_seed, NTUDatasetList

if __name__ == '__main__':
    log('Using {} CPU(s)'.format(torch.get_num_threads()))
    log('Using {} GPU(s)'.format(torch.cuda.device_count()))

    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Parse arguments
    args = argument_parser()

    # Initialize random seeds
    random_seed(args.seed)

    # Start a new wandb run
    wandb.init(project='UHAR_Skeletal_Laplacian', entity=args.wandb_user,
               name='{} {}'.format(args.method, args.split), config=args)

    # Enable SSVI if requested from method argument
    args.SSVI = False
    rotate_data = None
    if args.method == 'GRAE_L':
        args.SSVI = True
        rotate_data = RotationZ()

    # Initialize model with dataparallel (according to the # of GPUs available)
    model = torch.nn.DataParallel(GRAE_L(args).cuda(),
                                  device_ids=range(torch.cuda.device_count()))

    # Save model inputs and hyperparameters
    config = wandb.config

    # Log gradients and model parameters
    wandb.watch(model)

    # Arrange hyperparameters according to respective datasets
    if '60' in args.split:
        split_folder = 'NTU_60'
        lr = 1e-4
        Rskel_penalty = 1e-3
    elif '120' in args.split:
        split_folder = 'NTU_120'
        lr = 1e-3
        Rskel_penalty = 1e-4

    # Initialize the 1NN classifier
    knn = KNeighborsClassifier(n_neighbors=1, weights='uniform',
                               metric='cosine', n_jobs=-1)

    # Enable Rskel if requested from method argument
    if args.method == 'AE':
        laploss = None
        optimizer = torch.optim.Adam(model.parameters(), lr)
    else:
        laploss = LaplacianRegularizer(Rskel_penalty).cuda()
        optimizer = torch.optim.Adam(list(model.parameters()) +
                                     list(laploss.parameters()), lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.1,
                                                           patience=8,
                                                           cooldown=0,
                                                           verbose=True)

    # Load data
    if args.split == 'xsub60' or args.split == 'xsub120':
        log('Loading {} Cross-Subject Data'.format(split_folder))
        path = '{}/{}/xsub'.format(args.data_path, split_folder)
    elif args.split == 'xview60':
        log('Loading {} Cross-View Data'.format(split_folder))
        path = '{}/{}/xview'.format(args.data_path, split_folder)
    elif args.split == 'xset120':
        log('Loading {} Cross-Setup Data'.format(split_folder))
        path = '{}/{}/xset'.format(args.data_path, split_folder)

    train_data = torch.load('{}/train_data_1.pt'.format(path))
    train_label = torch.load('{}/train_label_1.pt'.format(path))
    test_data = torch.load('{}/test_data_1.pt'.format(path))
    test_label = torch.load('{}/test_label_1.pt'.format(path))

    train_dataset = NTUDatasetList(tensors=(train_data, train_label))
    test_dataset = NTUDatasetList(tensors=(test_data, test_label))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              shuffle=False)

    # Start the training
    log('Begin training')
    for epoch in range(args.epochs):
        log('--------------------')
        start = time()

        train_log = train(args=args, classifier=knn, dataloader=train_loader,
                          epoch=epoch, model=model, opt=optimizer,
                          rotate=rotate_data, rskel=laploss)
        test_log = test(args=args, classifier=knn, dataloader=test_loader,
                        epoch=epoch, model=model, rotate=rotate_data,
                        rskel=laploss)

        log('--------------------')
        log('Train Loss: {:.4f}'.format(train_log['Train Loss']))
        if args.method != 'AE':
            log('Train MSE: {:.4f}'.format(train_log['Train MSE']))
        if args.method == 'AE_L':
            log('Train Rskel: {:.4f}'.format(train_log['Train Rskel']))
        if args.method == 'GRAE_L':
            log('Train SSVI: {:.4f}'.format(train_log['Train SSVI']))

        log('--------------------')
        log('Test Loss: {:.4f}'.format(test_log['Test Loss']))
        if args.method != 'AE':
            log('Test MSE: {:.4f}'.format(test_log['Test MSE']))
        if args.method == 'AE_L':
            log('Test Rskel: {:.4f}'.format(test_log['Test Rskel']))
        if args.method == 'GRAE_L':
            log('Test SSVI: {:.4f}'.format(test_log['Test SSVI']))

        log('--------------------')
        log('Epoch runtime: {:.2f} s'.format(time() - start))
        log('Test Accuracy {:.2f}%'.format((100. * test_log['Test Accuracy'])))

        wandb.log({'Epochs': epoch + 1,
                   'Train Loss': train_log['Train Loss'],
                   'Test Loss': test_log['Test Loss'],
                   'Test Accuracy': test_log['Test Accuracy']})
        if args.method != 'AE':
            wandb.log({'Train MSE Loss': train_log['Train MSE'],
                       'Test MSE Loss': test_log['Test MSE']})
        if args.method == 'AE_L':
            wandb.log({'Train Rskel Loss': train_log['Train Rskel'],
                       'Test Rskel Loss': test_log['Test Rskel']})
        if args.method == 'GRAE_L':
            wandb.log({'Train SSVI Loss': train_log['Train SSVI'],
                       'Test SSVI Loss': test_log['Test SSVI']})

        scheduler.step(train_log['Train Loss'])

    log('Done!')
