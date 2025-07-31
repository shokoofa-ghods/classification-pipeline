"""
Main entry point
"""

import sys
import os
import glob

import torch
import random
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from matplotlib.axes import Axes

# pylint is unable to resolve the imports but they should be fine
from data import (video_aug_pipeline,
                  image_aug_pipeline,
                  CustomDatasetImg,
                  CustomDatasetVideo
                )
from common.constant import VIEW_MAP
                 # pylint: disable=import-error
from utils import (normalized_img_visualization,
                   show_learning_curve,
                   return_last_file,
                   show_distribution,
                   assert_exists,
                   assert_is_file,
                   assert_is_dir,
                   modify_view_map)               # pylint: disable=import-error
from models import (TransformerBased_video,
                    Spatial_Temporal,
                    MultiViewClassifier,
                    CNN_CBAM,
                    CNNLSTM,
                    EpochInfo,
                    model_type,
                    Trainer)                  # pylint: disable=import-error
from params import get_params                   # pylint: disable=import-error

REQUIRED_PARAMS = ['file_address',
                   'data_address',
                   'train',
                   'eval',
                   'multiframe']


def main() -> int:
    """
    Main function

    Returns:
        The status code to exit with
    """
    # Get program parameters
    try:
        params = get_params(sys.argv[1:], REQUIRED_PARAMS)
    except ValueError as e:
        print(e)
        return 1

    # For non-vital prints
    def log(*args, **kwargs):
        if params.verbose:
            print(*args, **kwargs)

    # log('params:', params, sep='\n')

    # Paths to csv files
    train_path = os.path.join(params.file_address, params.train_info_csv)
    val_path = os.path.join(params.file_address, params.val_info_csv)
    test_path = os.path.join(params.file_address, params.test_info_csv)

    # Validation
    try:
        # Check path existence
        assert_exists(params.file_address)
        assert_exists(train_path)
        assert_exists(val_path)
        assert_exists(test_path)
        # Check correct type and permissions
        assert_is_dir(params.file_address)
        assert_is_file(train_path, readable=True)
        assert_is_file(val_path, readable=True)
        assert_is_file(test_path, readable=True)
    except OSError as e:
        print(e)
        return 1

    def log_tensorboard(epoch:EpochInfo, writer:SummaryWriter, eps:float=1E-9):
        """
        Log an epoch to tensorboard

        Args:
            epoch (EpochInfo): The info about the epoch
            writer (SummaryWriter): The writer to use
            eps (float optional default: 1e-9): Small float to prevent 
                divide by zero
        """
        res = {
            key: {
                'true_pos': 0,
                'false_pos': 0,
                'true_neg': 0,
                'false_neg': 0
            } for key in params.VIEW_MAP
        }
        # Consider moving to train to save computation
        for i in range(len(epoch.eval['indices'])):
            for label, _ in enumerate(VIEW_MAP):
                if label == epoch.eval['predicted'][i]:
                    if epoch.eval['true_labels'][i] == label:
                        res[label]['true_pos'] += 1
                    else:
                        res[label]['false_pos'] += 1
                else:
                    if epoch.eval['true_labels'][i] == label:
                        res[label]['false_neg'] += 1
                    else:
                        res[label]['true_neg'] += 1
        run_f1 = 0
        for label, _ in enumerate(params.VIEW_MAP):
            all_pos = res[label]['true_pos'] + res[label]['false_neg']
            labeled_pos = res[label]['true_pos'] + res[label]['false_pos']
            recall = res[label]['true_pos'] / (all_pos + eps)
            precision = res[label]['true_pos'] / (labeled_pos + eps)
            f1 = 2 * (recall * precision) / (recall + precision + eps)
            run_f1 += f1
            writer.add_scalar(f'Recall/{label}', recall, epoch.num)
            writer.add_scalar(f'Precision/{label}', precision, epoch.num)
            writer.add_scalar(f'F1/{label}', f1, epoch.num)
        writer.add_scalar('Loss', epoch.loss, epoch.num)
        writer.add_scalar('Accuracy', epoch.acc, epoch.num)
        writer.add_scalar('Accuracy', epoch.acc, epoch.num)
        writer.add_scalar('Average_F1', run_f1 / len(params.VIEW_MAP), epoch.num)
    
    # Number of all samples per patients
    paths = glob.glob(os.path.join(params.data_address + '**/*/*', '*'))
    log(" Number of all video files:",len(paths))

    # Data Info
    info_train = pd.read_csv(train_path)
    info_val = pd.read_csv(val_path)
    info_test = pd.read_csv(test_path)
    log("Labels BEFORE Filter: ", info_test.label.unique())

    info_train, info_val, info_test = modify_view_map(info_train, info_val, info_test, params.combine_list)

    log("Labels AFTER Filter: ", info_test.label.unique())


    # Save Data
    subdir = 'src/saved_models/multiframe' if params.is_multiframe else 'src/saved_models/singleframe'
    save_dir_parent = os.path.join(params.file_address, subdir)
    assert_exists(save_dir_parent)
    save_dir = os.path.join(save_dir_parent, str(return_last_file(save_dir_parent)))
    os.makedirs(save_dir, exist_ok=True)

    # VIDEO_BASED
    if params.is_multiframe: 
        train_data = CustomDatasetVideo(info_train,
                                params.data_address,
                                params.VIEW_MAP,
                                params.data_mean,
                                params.data_std,
                                use_npy=False,
                                remove_ecg=True,
                                remove_static=True,
                                VIEW_MAP =params.VIEW_MAP
                                )
        val_data = CustomDatasetVideo(info_val,
                                params.data_address,
                                params.VIEW_MAP,
                                params.data_mean,
                                params.data_std,
                                use_npy=False,
                                remove_ecg=True,
                                remove_static=True,
                                )
    # IMAGE_BASED
    else:
        train_data = CustomDatasetImg(info_train,
                                params.data_address,
                                params.VIEW_MAP,
                                params.data_mean,
                                params.data_std,
                                use_npy=False,
                                remove_ecg=True,
                                remove_static=True,
                                )
        val_data = CustomDatasetImg(info_val,
                                params.data_address,
                                params.VIEW_MAP,
                                params.data_mean,
                                params.data_std,
                                use_npy=False,
                                remove_ecg=True,
                                remove_static=True,
                                )
        
    # Test visualize
    if params.plot:
        log('Data Distribution')
        show_distribution(info_train,info_val,info_test, save_dir)
        log('Visualizing Samples')
        num = random.randint(0, len(info_train))
        dt, lb = train_data[num]
        if params.is_multiframe:
            for i in range(5): #at least 5 frames will be in multiframe
                normalized_img_visualization(dt[i,:,:], lb, save_dir,i, params.data_mean, params.data_std)
        else:
            normalized_img_visualization(dt[0,:,:], lb, save_dir,0, params.data_mean, params.data_std)

    # Data Loaders
    log('Initializing Data Loaders')
    train_loader = DataLoader(train_data,
                              batch_size=params.batch_size,
                              drop_last=False,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=4,
                              persistent_workers=True)
    val_loader = DataLoader(val_data,
                            batch_size=params.batch_size,
                            drop_last=False,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4,
                            persistent_workers=True)
    if params.verbose:
        log('test read')
        for img, _ in train_loader:
            log(img.shape)
            break

    log('Initializing Model')
    # Model Initialization
    

    # Backbone Names : 'efficientnet-b2', 'efficientnet-b0', 'resnet18', 'convnext_small', 'convnext_base', 'coatnet'
    # Model Names : (video) TransformerBased_video, Spatial_Temporal, VideoCNNLSTM, (image) MultiViewClassifier, CNN_CBAM,
    

    model_type_dict = model_type['resnet18']
    model = CNN_CBAM(model_type_dict, len(params.VIEW_MAP))

    if params.base_model:
        print(f'Loading Model from "{params.base_model}"')
        state_dict = torch.load(os.path.join(params.file_address, params.base_model))
        model.load_state_dict(state_dict)
    # # Test model
    # if params.verbose:
    #     model.eval()
    #     log(model(torch.rand(8, 3, 299, 299))[0].shape)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    log(f'Using device: {device}')

    if torch.cuda.device_count() > 1:
        log(f'Let\'s use {torch.cuda.device_count()} GPUs!')
        model = nn.DataParallel(model)

    # Training init
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    if params.tensorboard:
        writer = SummaryWriter()
        trainer = Trainer(criterion,
                          params.epochs,
                          device,
                          no_improve_limit=params.loss_limit,
                          after_n_epochs_no_improve_limit= params.after_n_epochs,
                          log_epoch=lambda e:log_tensorboard(e, writer))
    else:
        trainer = Trainer(criterion,
                          params.epochs,
                          device,
                          no_improve_limit=params.loss_limit, 
                          after_n_epochs_no_improve_limit= params.after_n_epochs)
    
    _accs, _losses = trainer(model,
                           optimizer,
                           train_loader,
                           val_loader,
                           save_dir,
                           params.VIEW_MAP,
                           verbose=params.verbose
                           )
    
    show_learning_curve(accs=_accs, losses=_losses, dir=save_dir)   


if __name__ == '__main__':
    sys.exit(main())
