# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import numpy as np
import matplotlib.pyplot as plt

from ..DCGAN import DCGAN
from .gan_trainer import GANTrainer
from .standard_configurations.dcgan_config import _C


class DCGANTrainer(GANTrainer):
    r"""
    A trainer structure for the DCGAN and DCGAN product models
    """

    _defaultConfig = _C

    def getDefaultConfig(self):
        return DCGANTrainer._defaultConfig

    def __init__(self,
                 pathdb,
                 **kwargs):
        r"""
        Args:

            pathdb (string): path to the input dataset
            **kwargs:        other arguments specific to the GANTrainer class
        """

        GANTrainer.__init__(self, pathdb, **kwargs)

        self.lossProfile.append({"iter": [], "scale": 0})
        self.G_loss, self.D_loss = 0, 0
        self.G_losses, self.D_losses, self.y_list = [], [], []



    def initModel(self):
        self.model = DCGAN(useGPU=self.useGPU,
                           **vars(self.modelConfig))

    def train(self):

        shift = 0
        if self.startIter >0:
            shift+= self.startIter

        if self.checkPointDir is not None:
            pathBaseConfig = os.path.join(self.checkPointDir, self.modelLabel
                                          + "_train_config.json")
            self.saveBaseConfig(pathBaseConfig)

        maxShift = int(self.modelConfig.nEpoch * len(self.getDBLoader(0)))
        self.D_loss = 0
        self.G_loss = 0

        #optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
        

        for epoch in range(self.modelConfig.nEpoch):
            print("\nstarting epoch",epoch)
            dbLoader = self.getDBLoader(0)
            
            _, self.G_loss, self.D_loss = self.trainOnEpoch(dbLoader, 0, shiftIter=shift)
            self.D_losses.append(self.D_loss)
            self.G_losses.append(self.G_loss)

            shift += len(dbLoader)

            if shift > maxShift:
                break

        ##### グラフ作成＆保存 #####
        #print(self.G_losses)
        #print(self.D_losses)
        self.np_G_losses = np.array(self.G_losses)
        self.np_D_losses = np.array(self.D_losses)
        self.y_list = np.array(list(range(1,len(self.np_D_losses)+1)))

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax1.set_title('Generator Loss')
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        self.ax2.set_title('Dicsriminator Loss')

        
        self.ax1.plot(self.y_list, self.np_G_losses)
        plt.ylim(0, 5)
        self.ax2.plot(self.y_list, self.np_D_losses)
        plt.ylim(0, 1)
        self.out_img_path = os.path.join("output_networks","loss_img")
        if not os.path.isdir(self.out_img_path):
            os.mkdir(self.out_img_path)

        self.fig.savefig(os.path.join(self.out_img_path, 'test1_2.jpg'))


        label = self.modelLabel + ("_s%d_i%d" %
                                   (0, shift))
        self.saveCheckpoint(self.checkPointDir,
                            label, 0, shift)

    def initializeWithPretrainNetworks(self,
                                       pathD,
                                       pathGShape,
                                       pathGTexture,
                                       finetune=True):
        r"""
        Initialize a product gan by loading 3 pretrained networks

        Args:

            pathD (string): Path to the .pt file where the DCGAN discrimator is saved
            pathGShape (string): Path to .pt file where the DCGAN shape generator
                                 is saved
            pathGTexture (string): Path to .pt file where the DCGAN texture generator
                                   is saved

            finetune (bool): set to True to reinitialize the first layer of the
                             generator and the last layer of the discriminator
        """

        if not self.modelConfig.productGan:
            raise ValueError("Only product gan can be cross-initialized")

        self.model.loadG(pathGShape, pathGTexture, resetFormatLayer=finetune)
        self.model.load(pathD, loadG=False, loadD=True,
                        loadConfig=False, finetuning=True)
