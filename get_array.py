import csv
import time
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score



from data.data_entry import select_loader, get_labels
from options import prepare_train_args

import loss

def label_abstract_reduce(labels, abstract_ids):
    # example: val_preds_abstract = label_abstract_reduce(val_preds, val_ids)
    # example: val_gt_abstract = label_abstract_reduce(val_gts, val_ids)

    result = defaultdict(lambda:0)
    unique_abs_id = sorted(list(set(abstract_ids)))  # abstract_ids以不同顺序传入时均可使用

    for abstract_id, label in zip(abstract_ids, labels):
        result[abstract_id] |= label

    return np.stack([result[abstract] for abstract in unique_abs_id])

args = prepare_train_args()

train_text, train_labels, train_ids = get_labels(args, mode='train')
dev_text, dev_labels, dev_ids = get_labels(args, mode='dev')
test_text, test_labels, test_ids = get_labels(args, mode='test')

