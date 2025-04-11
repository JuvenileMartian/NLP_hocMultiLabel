import csv
import time
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.autograd import Variable

from data.data_entry import select_loader
from model.model_entry import select_model
from options import prepare_train_args
from utils.logger import Logger
from utils.torch_utils import load_match_dict
from transformers import BertModel, BertTokenizer, get_scheduler, AutoModel
from tqdm import tqdm

import loss

def compute_multilabel_f1(gt, pred):
    # gt: n*10
    # pred: n*10

    gt = gt.astype(int)
    pred = pred.astype(int)

    tp = gt & pred

    precision = (tp.sum(1) / np.where(pred.sum(1) < 1, 1, pred.sum(1))).mean()
    recall = (tp.sum(1) / np.where(gt.sum(1) < 1, 1, gt.sum(1))).mean()

    return 2 * precision * recall / (precision + recall)


class Trainer:
    def __init__(self):
        args = prepare_train_args()
        self.args = args
        #torch.manual_seed(args.seed)
        self.logger = Logger(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")

        self.train_loader = select_loader(args, mode='train')
        self.val_loader = select_loader(args, mode='dev')
        self.test_loader = select_loader(args, mode='test')

        self.model = select_model(args)
        if args.load_model_path == '':
            print("Using pretrained model")
        else:
            self.model.load_state_dict(torch.load(args.load_model_path).state_dict())
        self.model.to(self.device)

        #self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_fn = loss.FocalLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        self.scheduler = get_scheduler("linear", optimizer=self.optimizer, num_warmup_steps=50, num_training_steps=len(self.train_loader)*args.epochs)



    def train(self):
        for epoch in range(self.args.epochs):
            # train for one epoch
            self.train_per_epoch(epoch)
            self.validate_dataset(epoch, self.train_loader)
            self.val_per_epoch(epoch)
            self.test_per_epoch(epoch)
            self.logger.save_curves(epoch)
            self.logger.save_check_point(self.model, epoch)

    def train_per_epoch(self, epoch):
        # switch to train mode
        #scaler = torch.amp.GradScaler("cuda")
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Training Epoch {epoch+1}")
        for i, data in progress_bar:
            logits, gt_labels = self.step(data)

            # compute loss
            loss = self.loss_fn(logits, gt_labels)

            # compute gradient and do Adam step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # logger record
            self.logger.record_scalar('train/loss', loss)

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss:.6f}")

        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    def validate_dataset(self, epoch, loader):
        THRESHOLD = 0.5
        self.model.eval()
        total_loss = 0
        val_preds = []
        val_ids = []
        val_gts = []
        with torch.no_grad():
            progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Validation for Epoch {epoch+1}")
            for i, data in progress_bar:
                logits, gt_labels = self.step(data)
                loss = self.loss_fn(logits, gt_labels)

                self.logger.record_scalar('validation/loss', loss)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss:.4f}")

                pred_labels = (torch.sigmoid(logits) > THRESHOLD).int()
                val_preds.append(pred_labels.cpu().numpy())
                val_gts.append(gt_labels.cpu().int().numpy())
                val_ids += data["abstract_ids"]
                

        val_preds = np.concatenate(val_preds)
        val_gts = np.concatenate(val_gts)
        unique_abs_id = set(val_ids)

        val_preds_abstract = label_abstract_reduce(val_preds, val_ids)
        val_preds_abstract = np.stack([val_preds_abstract[abstract] for abstract in unique_abs_id])
        val_gt_abstract = label_abstract_reduce(val_gts, val_ids)
        val_gt_abstract = np.stack([val_gt_abstract[abstract] for abstract in unique_abs_id])

        score = f1_score(val_gt_abstract, val_preds_abstract , average='samples')
        print(score)
        #self.logger.record_scalar('validation/f1', score)
        #im_arr = self.logger.gt_pred_img(val_gt_abstract, val_preds_abstract, epoch)
        #self.logger.save_eval_img('Dev', im_arr, epoch)


    def val_per_epoch(self, epoch):
        THRESHOLD = 0.5
        self.model.eval()
        total_loss = 0
        val_preds = []
        val_ids = []
        val_gts = []
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc=f"Validation for Epoch {epoch+1}")
            for i, data in progress_bar:
                logits, gt_labels = self.step(data)
                loss = self.loss_fn(logits, gt_labels)

                self.logger.record_scalar('validation/loss', loss)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss:.4f}")

                pred_labels = (torch.sigmoid(logits) > THRESHOLD).int()
                val_preds.append(pred_labels.cpu().numpy())
                val_gts.append(gt_labels.cpu().int().numpy())
                val_ids += data["abstract_ids"]
                

        val_preds = np.concatenate(val_preds)
        val_gts = np.concatenate(val_gts)
        unique_abs_id = set(val_ids)

        val_preds_abstract = label_abstract_reduce(val_preds, val_ids)
        val_preds_abstract = np.stack([val_preds_abstract[abstract] for abstract in unique_abs_id])
        val_gt_abstract = label_abstract_reduce(val_gts, val_ids)
        val_gt_abstract = np.stack([val_gt_abstract[abstract] for abstract in unique_abs_id])

        score = f1_score(val_gt_abstract, val_preds_abstract , average='samples')
        print(score)
        self.logger.record_scalar('validation/f1', score)
        im_arr = self.logger.gt_pred_img(val_gt_abstract, val_preds_abstract, epoch)
        #self.logger.save_eval_img('Dev', im_arr, epoch)

    def test_per_epoch(self, epoch):
        # running val_per_epoch on test set
        THRESHOLD = 0.5
        self.model.eval()
        total_loss = 0
        val_preds = []
        val_ids = []
        val_gts = []
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc=f"Validation for Epoch {epoch+1}")
            for i, data in progress_bar:
                logits, gt_labels = self.step(data)
                loss = self.loss_fn(logits, gt_labels)

                self.logger.record_scalar('test/loss', loss)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss:.4f}")

                pred_labels = (torch.sigmoid(logits) > THRESHOLD).int()
                val_preds.append(pred_labels.cpu().numpy())
                val_gts.append(gt_labels.cpu().int().numpy())
                val_ids += data["abstract_ids"]
                

        val_preds = np.concatenate(val_preds)
        val_gts = np.concatenate(val_gts)
        unique_abs_id = set(val_ids)

        val_preds_abstract = label_abstract_reduce(val_preds, val_ids)
        val_preds_abstract = np.stack([val_preds_abstract[abstract] for abstract in unique_abs_id])
        val_gt_abstract = label_abstract_reduce(val_gts, val_ids)
        val_gt_abstract = np.stack([val_gt_abstract[abstract] for abstract in unique_abs_id])

        score = f1_score(val_gt_abstract, val_preds_abstract , average='samples')
        print(score)
        self.logger.record_scalar('test/f1', score)

        #im_arr = self.logger.gt_pred_img(val_gt_abstract, val_preds_abstract, epoch)
        #self.logger.save_eval_img('Test', im_arr, epoch)




    def step(self, data):
        input_ids = data["input_ids"].to(self.device)
        attention_mask = data["attention_mask"].to(self.device)
        labels = data["labels"].to(self.device)
        logits = self.model(input_ids, attention_mask)
        return logits, labels

    def compute_f1(self, pred, gt, is_train):
        pass


def label_abstract_reduce(labels, abstract_ids):
    result = defaultdict(lambda:0)
    for abstract_id, label in zip(abstract_ids, labels):
        result[abstract_id] |= label
    return result

def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
