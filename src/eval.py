"""
Evaluation program
"""

import sys
import os
import glob
import re

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from data import CustomDataset, video_aug_pipeline                         # pylint: disable=import-error
from models import ImageCNNLSTM, VideoCNNLSTM, Trainer                      # pylint: disable=import-error
from utils import VALID_LABELS, show_confusion_matrix, calculate_accuracy   # pylint: disable=import-error
from params import get_params                                               # pylint: disable=import-error

REQUIRED_PARAMS = ['test',
                   'multiframe'
]

def _get_dict_index(path:str) -> int:
    match = re.match(r'.*test([0-9]+).pth', path)
    if match is None:
        return -1
    return int(match.group(1))

REQUIRED_PARAMS = [
    'test',
    'origin',
    'multiframe'
]

def main() -> int:
    """
    Main entry point for eval
    """

    params = get_params(sys.argv[1:], REQUIRED_PARAMS)
    is_multiframe = params.is_multiframe
    origin = params.original_address
    epochs = params.epochs

    info_test = pd.read_csv(os.path.join(origin, params.test_info_csv))
    test_data = CustomDataset(info_test['path'],
                              info_test['label'],
                              video_aug_pipeline,
                              origin,
                              frames=3)
    test_loader = DataLoader(test_data,
                             batch_size=32,
                             drop_last=False,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=8,
                             persistent_workers=True)

    device = 'cpu'#'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'using device {device}')
    if is_multiframe:
        model = VideoCNNLSTM()
        state_dict_paths = glob.glob(os.path.join(origin, 'multiframe', 'test*.pth'))
    else:
        model = ImageCNNLSTM()
        state_dict_paths = glob.glob(os.path.join(origin, 'singleframe', 'test*.pth'))
    if params.base_model:
        best_dict = os.path.join(params.original_address, params.base_model)
    else:
        best_dict = sorted(state_dict_paths, key=_get_dict_index)[-1]
    print(f'Loading Model from "{best_dict}"...')
    state_dict = torch.load(best_dict)
    model.load_state_dict(state_dict)
    print('Model Loaded')
    model.to(device)
    #model = nn.DataParallel(model, device_ids = [0, 1])

    trainer = Trainer(nn.CrossEntropyLoss(),
                      epochs,
                      device=device)
    _, _, test_dict = trainer.evaluate(model,
                     test_loader,
                     advanced=True,
                     mode='Test')
    test_df = pd.DataFrame(test_dict)
    test = pd.concat([test_df, info_test], axis=1)
    print(test)


    def get_mode(series):
        return series.mode()[0] if not series.mode().empty else None
    res = test.groupby(['path', 'true_labels']).agg({'predicted': get_mode}).reset_index()
    print(res)

    print(res[res['true_labels'] != res['predicted']])
    true_labels, pred_labels = res['true_labels'], res['predicted']

    print(calculate_accuracy(true_labels, pred_labels))

    labels_name = VALID_LABELS.keys()
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    show_confusion_matrix(cm, labels_name, fmt='d')
    show_confusion_matrix(cm_normalized, labels_name,)

    return 0


if __name__ == '__main__':
    sys.exit(main())
