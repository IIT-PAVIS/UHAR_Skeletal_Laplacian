"""
    Training and testing routines.

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

from tqdm import tqdm
from torch.nn.functional import mse_loss, l1_loss

from classifier import knn_classification
from utils import Metric


def train(args, classifier, dataloader, epoch, model, rotate, opt, rskel):
    model.train()
    feature, label = torch.empty(0), torch.empty(0)

    # Initialize metrics
    trainLoss = Metric('trainLoss')
    if args.method != 'AE':
        trainMSE = Metric('trainMSE')
    if args.method == 'AE_L':
        trainRskel = Metric('trainRskel')
    if args.method == 'GRAE_L':
        trainSSVI = Metric('trainSSVI')

    # Start batch training
    infos = 'Train {} epoch {}/{}'.format(args.method, epoch + 1, args.epochs)
    for data, batch_label in tqdm(dataloader, desc=infos):
        data = data.float().cuda()

        # Compute AE method
        if args.method == 'AE':
            data_hat, batch_feature = model(data)
            mseLoss = mse_loss(data, data_hat)
            batch_loss = mseLoss

        # Compute AE_L method
        elif args.method == 'AE_L':
            data_hat, batch_feature = model(data)
            mseLoss = mse_loss(data, data_hat)
            rskelLoss = rskel(data_hat)
            batch_loss = mseLoss + rskelLoss

        # Compute GRAE_L method
        elif args.method == 'GRAE_L':
            rotated_data, alpha_rot = rotate(data)
            data_hat, batch_feature, alpha_rot_hat = model(rotated_data)
            ssviLoss = (args.SSVI_penalty * l1_loss(torch.tensor(alpha_rot),
                                                    alpha_rot_hat.squeeze()))
            mseLoss = mse_loss(rotated_data, data_hat)
            rskelLoss = rskel(data_hat)
            batch_loss = ssviLoss + mseLoss + rskelLoss

        # Update metrics
        trainLoss.update(batch_loss)
        if args.method != 'AE':
            trainMSE.update(mseLoss)
        if args.method == 'AE_L':
            trainRskel.update(rskelLoss)
        if args.method == 'GRAE_L':
            trainSSVI.update(ssviLoss)

        opt.zero_grad()
        batch_loss.div_(math.ceil(float(len(data)) / args.batch_size))
        batch_loss.backward()
        opt.step()

        # Store features from latent space and labels for classification
        feature = torch.cat((feature, batch_feature.cpu()), dim=0)
        label = torch.cat((label, batch_label), dim=0)

    model.mean = torch.mean(feature, dim=0)
    model.std = torch.std(feature, dim=0)
    features = ((feature - model.mean) / model.std).detach().numpy()
    labels = label.detach().numpy()

    # Classify with knn
    knn_classification(classifier, features, labels, step='train')

    # Log metrics
    train_log = {'Train Loss': trainLoss.avg.item()}
    if args.method != 'AE':
        train_log['Train MSE'] = trainMSE.avg.item()
    if args.method == 'AE_L':
        train_log['Train Rskel'] = trainRskel.avg.item()
    if args.method == 'GRAE_L':
        train_log['Train SSVI'] = trainSSVI.avg.item()

    return train_log


def test(args, classifier, dataloader, epoch, model, rotate, rskel):
    model.eval()
    feature, label = torch.empty(0), torch.empty(0)

    # Initialize metrics
    testLoss = Metric('testLoss')
    if args.method != 'AE':
        testMSE = Metric('testMSE')
    if args.method == 'AE_L':
        testRskel = Metric('testRskel')
    if args.method == 'GRAE_L':
        testSSVI = Metric('testSSVI')

    # Start batch testing
    infos = 'Test  {} epoch {}/{}'.format(args.method, epoch + 1, args.epochs)
    for data, batch_label in tqdm(dataloader, desc=infos):
        data = data.float().cuda()
        with torch.no_grad():

            # Compute AE method
            if args.method == 'AE':
                data_hat, batch_feature = model(data)
                mseLoss = mse_loss(data, data_hat)
                batch_loss = mseLoss

            # Compute AE_L method
            elif args.method == 'AE_L':
                data_hat, batch_feature = model(data)
                mseLoss = mse_loss(data, data_hat)
                rskelLoss = rskel(data_hat)
                batch_loss = mseLoss + rskelLoss

            # Compute GRAE_L method
            elif args.method == 'GRAE_L':
                rotated_data, alpha_rot = rotate(data)
                data_hat, batch_feature, alpha_rot_hat = model(rotated_data)
                ssviLoss = (args.SSVI_penalty * l1_loss(torch.tensor(alpha_rot),
                                                        alpha_rot_hat.squeeze()))
                mseLoss = mse_loss(rotated_data, data_hat)
                rskelLoss = rskel(data_hat)
                batch_loss = ssviLoss + mseLoss + rskelLoss

        # Update metrics
        testLoss.update(batch_loss)
        if args.method != 'AE':
            testMSE.update(mseLoss)
        if args.method == 'AE_L':
            testRskel.update(rskelLoss)
        if args.method == 'GRAE_L':
            testSSVI.update(ssviLoss)

        # Store features from latent space and labels for classification
        feature = torch.cat((feature, batch_feature.cpu()), dim=0)
        label = torch.cat((label, batch_label), dim=0)

    features = ((feature - model.mean) / model.std).detach().numpy()
    labels = label.detach().numpy()

    # Classify with knn
    testAcc = knn_classification(classifier, features, labels, step='test')

    # Log metrics
    test_log = {'Test Loss': testLoss.avg.item(),
                'Test Accuracy': testAcc}
    if args.method != 'AE':
        test_log['Test MSE'] = testMSE.avg.item()
    if args.method == 'AE_L':
        test_log['Test Rskel'] = testRskel.avg.item()
    if args.method == 'GRAE_L':
        test_log['Test SSVI'] = testSSVI.avg.item()

    return test_log
