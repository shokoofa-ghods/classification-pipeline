"""
Data visualization module
"""

import matplotlib.pyplot as plt
import torch
import seaborn as sns
import os
import pandas as pd
from matplotlib.axes import Axes
from sklearn.metrics import confusion_matrix
import numpy as np


from common.constant import VIEW_MAP

def img_visualization(image:torch.Tensor, label:torch.Tensor, dir:str):
    """
    Plot an image with label
    
    Args:
        image (Image): The image to display
        label (str): The title for the image
    """
    plt.figure(figsize= (8,8))
    # im = Image.fromarray(image.permute(1,2,0))
    # plt.imshow(image.permute(1,2,0))
    plt.imshow(image, cmap= 'gray')
    plt.title(str(label.item()))
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(dir,'image_sample.png'))


def normalized_img_visualization(image, label, dir, i, mean, std):
    """
    Plot and convert images that were normalized

    Args:
        image (Image): the image to display
        label (str): the title for the image
        mean (list[float]): list of mean per channel
        std (list[float]): list of std per channel
        i (int): index number of the image frame in video
    """
    # Un-normalize
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    image = image * std + mean  # reverse normalization

    # Convert to [0, 1] and clamp to avoid display issues
    image = torch.clamp(image, 0, 1)

    # Convert to HWC
    image_np = image.permute(1, 2, 0).cpu().numpy()

    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(image_np)
    plt.title(f"Label: {label.item()}")
    plt.axis('off')
    plt.savefig(os.path.join(dir,f'image_sample{i}.png'))


def show_confusion_matrix(true_labels: list,
                          pred_labels: list, 
                          dir: str,
                          count_cm:bool = True,
                          epoch_num :int = -1,
                          title:str='Confusion Matrix',
                          x_label:str='Predicted Labels',
                          y_label:str='True Labels',
                          fmt:str='.2g',
                          cmap:str='Blues',
                          figsize:tuple[int, int]=(10, 7)) -> None:
    """
    Plot and show a confusion matrix along with its labels

    Args:
        true_labels (list[int]): list of true labels
        pred_labels (list[int]): list of predicted labels
        labels_name (Sequence[str]): The sequence of string labels 
    """
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if count_cm:        
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=VIEW_MAP, yticklabels=VIEW_MAP )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()

    plt.figure(figsize=(10,7))
    sns.heatmap(cm_normalized, annot=True, cmap='Blues', xticklabels=VIEW_MAP, yticklabels=VIEW_MAP )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    if epoch_num == -1:
        plt.savefig(os.path.join(dir, '(TEST)-confusion_matrix.png'))
    else:
        plt.savefig(os.path.join(dir, f'(epoch-{epoch_num})-confusion_matrix.png'))
    

def show_distribution(info_train, info_val, info_test, dir):
    train_dict = info_train['label'].value_counts().to_dict()
    val_dict =info_val['label'].value_counts().to_dict()
    test_dist_dict = info_test['label'].value_counts().to_dict()

    all_classes = set(train_dict.keys()).union(set(val_dict.keys()))
    train_dict_full = {cls: train_dict.get(cls, 0) for cls in all_classes}
    val_dict_full = {cls: val_dict.get(cls, 0) for cls in all_classes}
    test_dict_full = {cls: test_dist_dict.get(cls, 0) for cls in all_classes}

    df_counts = pd.DataFrame({
        'Train': train_dict_full,
        'Validation': val_dict_full,
        'Test': test_dict_full
    })

    df_counts = df_counts.sort_index()

    # df_counts = pd.DataFrame({'Train': train_counts, 'Validation': val_counts})

    ax = df_counts.plot(kind='bar', figsize=(14, 7))
    ax.set_xlabel("Class Name")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Classes in Training and Validation Sets")
    fig = ax.get_figure()
    fig.savefig(os.path.join(dir, 'distribution.png'))



def show_learning_curve(accs, losses, dir):
    """
    Plot and show a the learning curves of training and validation sets

    Args:
        accs (list) list of accuracies over train and validation 
        losses (list) list of losses over train and validation
    """
    x = [i for i in range(len(accs) // 2)]
    y = [accs[i].cpu().numpy() for i in range(0, len(accs), 2)]
    z = [accs[i].cpu().numpy() for i in range(1, len(accs), 2)]

    # Create a new figure and set the size
    fig = plt.figure(figsize=(8, 6))

    # Add a new subplot to the figure
    ax:Axes = fig.add_subplot(1, 1, 1)

    # Plot the line graph
    ax.plot(x, y, label='accuracy')
    ax.plot(x, z, label='val_accuracy')
    ax.legend()

    # Set the title and axis labels
    ax.set_title('learning accuracy curve')
    ax.set_xlabel('epochs')
    ax.set_ylabel('accuracy')
    fig.savefig(os.path.join(dir, 'learning_curve.png'))

    # Display the plot
    return 0
    
