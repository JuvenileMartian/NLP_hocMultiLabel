from tensorboardX import SummaryWriter
import os
import torch

import numpy as np
import matplotlib.pyplot as plt
import io
import seaborn as sns
import cv2


class Recoder:
    def __init__(self):
        self.metrics = {}

    def record(self, name, value):
        if name in self.metrics.keys():
            self.metrics[name].append(value)
        else:
            self.metrics[name] = [value]

    def summary(self):
        kvs = {}
        for key in self.metrics.keys():
            kvs[key] = sum(self.metrics[key]) / len(self.metrics[key])
            del self.metrics[key][:]
            self.metrics[key] = []
        return kvs


class Logger:
    def __init__(self, args):
        self.writer = SummaryWriter(args.model_dir)
        self.recoder = Recoder()
        self.model_dir = args.model_dir

    def tensor2img(self, tensor):
        # implement according to your data, for example call viz.py
        return tensor.cpu().numpy()

    def record_scalar(self, name, value):
        self.recoder.record(name, value)

    def save_curves(self, epoch):
        kvs = self.recoder.summary()
        for key in kvs.keys():
            self.writer.add_scalar(key, kvs[key], epoch)

    def save_imgs(self, names2imgs, epoch):
        for name in names2imgs.keys():
            self.writer.add_image(name, self.tensor2img(names2imgs[name]), epoch)

    def save_check_point(self, model, epoch, step=0):
        model_name = '{epoch:02d}_{step:06d}.pth'.format(epoch=epoch, step=step)
        path = os.path.join(self.model_dir, model_name)
        # don't save model, which depends on python path
        # save model state dict
        #torch.save(model.state_dict(), path)

    def gt_pred_img(self, gt, pred, epoch=None):
        

        gt = gt.T
        pred = pred.T
        m,n = gt.shape
    
        fig, ax = plt.subplots(figsize=(200,600))
        sns.heatmap(gt+2*pred, cmap=["white", "red", 'blue', 'green'], cbar=False, linewidths=0.5, linecolor='black', ax=ax)

        
        ax.set_xticks([])
        ax.set_yticks([])

        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        if epoch is not None:
            fig.savefig('./figs/'+str(epoch)+'.png')
        plt.close(fig)

        
        buf.seek(0)
        image_array = plt.imread(buf)

        
        return image_array
    
    def save_eval_img(self, name, img, epoch):
        resized_image = cv2.resize(img[:,:,:3], (512,512), interpolation=cv2.INTER_AREA)
        self.writer.add_image(name, resized_image, epoch, dataformats='HWC')
