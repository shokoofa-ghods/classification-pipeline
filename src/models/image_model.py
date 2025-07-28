"""
Module containing Image Model
"""

import torch.nn as nn
import torchvision
import torch
import timm

from common.constant import (INITIAL_LABELS,
                            COMBINED_VIEW_MAP,
                            VIEW_MAP)

NUM_CLASSES = len(VIEW_MAP) #8


# Two Stage View Classifier Model

# ----------- Router Model -------------- #
class Router(nn.Module):
    def __init__(self, feature_dim, group_to_labels):
        super().__init__()
        num_groups = len(group_to_labels)
        self.classifier = nn.Linear(feature_dim, num_groups)
        self.group_names2 = list(group_to_labels.keys()) #['apical', 'psax', 'other']

    def forward(self, features):
        return self.classifier(features) #[B, 4]


# ------------ Multi-Head Classifier ------------ #
class MultiHeadClassifier(nn.Module):
    def __init__(self, feature_dim, group_to_labels):
        super().__init__()
        self.group_to_labels = group_to_labels
        self.heads = nn.ModuleDict()
        for group, labels in group_to_labels.items():
            self.heads[group] = nn.Linear(feature_dim, len(labels))
        self.group_names2 = list(group_to_labels.keys()) #['apical', 'psax', 'other']

    def forward(self, features, group_probs):
        logits = torch.zeros(features.size(0), max(max(v) for v in self.group_to_labels.values()) + 1).to(features.device)
        for i, group in enumerate(self.group_names2):
            head = self.heads[group]
            group_logits = head(features)
            mapped_labels = self.group_to_labels[group]
            probs = group_probs[:, i].unsqueeze(1)  # (B, 1)
            for j, view_id in enumerate(mapped_labels):
                logits[:, view_id] += probs.squeeze() * group_logits[:, j]
        return logits


# ------------ Full Model ------------ #
class MultiViewClassifier(nn.Module):

    def __init__(self, model_type):
        super().__init__()

        self.view_to_group = {
        0: 'other',    # PLAX
        1: 'other',    # PSAX-ves
        2: 'psax',     # PSAX-sub
        3: 'apical',   # Apical-2ch
        4: 'apical',   # Apical-3ch
        5: 'apical',   # Apical-4&5ch
        6: 'other',   # Suprasternal
        7: 'other',    # Subcostal
        }

        self.group_to_labels = {
            'apical': [3,4,5],
            'psax': [2],
            'other': [0,1,6,7],
        }

        self.backbone_type = model_type['backbone']
        self.feature_channels = model_type['feature_channels']
        self.temporal_type = ''

        if self.backbone_type.startswith('convnext'):
            base_model = getattr(torchvision.models, self.backbone_type)(weights='IMAGENET1K_V1')

        elif self.backbone_type.startswith('resnet18'):
            base_model = torchvision.models.resnet18(weights='DEFAULT')
            self.features = nn.Sequential(*list(base_model.children())[:-2])

        elif self.backbone_type.startswith('efficientnet-b0'):
            backbone = torchvision.models.efficientnet_b0(weights='DEFAULT')

        elif self.backbone_type.startswith('efficientnet-b2'):
            base_model = torchvision.models.efficientnet_b2(weights='DEFAULT')
            
        else:
            base_model = timm.create_model(self.backbone_type, pretrained=True, features_only=True) ## gets 224x224

        self.features = nn.Sequential(base_model.features, nn.AdaptiveAvgPool2d((1, 1)))
        self.flatten = nn.Flatten()
        self.router = Router(feature_dim=self.feature_channels, group_to_labels=self.group_to_labels)
        self.multihead = MultiHeadClassifier(feature_dim=self.feature_channels, group_to_labels=self.group_to_labels)
        self.view_id_to_group_idx2 = {
            view_id: self.router.group_names2.index(self.view_to_group[view_id])
            for view_id in range(NUM_CLASSES)
        }
        self.group_names = VIEW_MAP #['plax', 'psax-ves', 'psax-sub', 'apical-2ch', 'apical-3ch', 'apical-4&5ch', 'suprasternal', 'subcostal']
        self.view_id_to_group_idx = {
            view_id: self.group_names.index(COMBINED_VIEW_MAP[view_id])
            for view_id in range(len(INITIAL_LABELS)) #11
        }


    def forward(self, x):
        x = self.features(x)  # (B, 768, 1, 1)
        x = self.flatten(x)   # (B, 768)
        group_logits = self.router(x)
        group_probs = torch.softmax(group_logits, dim=1)
        view_logits = self.multihead(x, group_probs)
        return view_logits, group_logits





# ------------------------------------------------------------------------------------------------------------
# CNN (Pre-trained Backbone e.g., ConvNext, Timm, EfficientNet, ResNet) + Channel and Spatial Attention Module

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)
        self.dropout2d = nn.Dropout2d(p=0.2)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        x = self.dropout2d(x)
        return x

class CNN_CBAM(nn.Module):
    def __init__(self, model_type ):
        super(CNN_CBAM, self).__init__()

        self.backbone_type = model_type['backbone']
        self.feature_channels = model_type['feature_channels']
        self.temporal_type = ''

        self.group_names = VIEW_MAP #['plax', 'psax-ves', 'psax-sub', 'apical-2ch', 'apical-3ch', 'apical-4&5ch', 'suprasternal', 'subcostal']
        self.view_id_to_group_idx = {
            view_id: self.group_names.index(COMBINED_VIEW_MAP[view_id])
            for view_id in range(len(INITIAL_LABELS)) #11
        }

        if self.backbone_type.startswith('convnext'):
            base_model = getattr(torchvision.models, self.backbone_type)(weights='IMAGENET1K_V1')
            self.features = base_model.features
            # self.feature_channels = 768 # small 768 # base 1024

        elif self.backbone_type.startswith('efficientnet-b2'):
            base_model = torchvision.models.efficientnet_b2(weights='DEFAULT')
            self.features = base_model.features
            # self.feature_channels = 1408
            
        else:
            base_model = timm.create_model(self.backbone_type, pretrained=True, features_only=True) ## gets 224x224
            self.features = base_model
            self.feature_channels = base_model.feature_info[-1]['num_chs']

        self.dropout2d = nn.Dropout2d(p=0.2)

        self.cbam = CBAM(channels=self.feature_channels)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_channels, NUM_CLASSES)
        )

    def forward(self, x):
        if isinstance(self.features, nn.Sequential ):
            x = self.features(x)
        else:
            x = self.features(x)[-1]

        x = self.dropout2d(x)
        # x = self.cbam(x)
        x = self.classifier(x)
        return x



# ------------------------------------------------------------------------------------------------------------
# EfficientNet + SE Model

# class SEBlock(nn.Module):
#     """
#     Squeeze Excitation Block
#     """
#     def __init__(self, input_channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Linear(input_channels, input_channels // reduction, bias=False)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(input_channels // reduction, input_channels, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         """
#         Forward Pass
#         """
#         batch_size, channels, _, _ = x.size()
#         y = self.global_avg_pool(x).view(batch_size, channels)
#         y = self.fc1(y)
#         y = self.relu(y)
#         y = self.fc2(y)
#         y = self.sigmoid(y).view(batch_size, channels, 1, 1)
#         return x * y

# class CNN(nn.Module):
#     """
#     Image CNN LSTM Model
#     """
#     def __init__(self, model_type):
#         super(CNN, self).__init__()

#         self.backbone_type = model_type['backbone']
#         self.feature_channels = model_type['feature_channels']
#         self.temporal_type = ''

#         # Load EfficientNet B2 without the final classification layer
#         self.conv = torchvision.models.efficientnet_b2(pretrained=True).features

#         # SE block after EfficientNet
#         self.se_block = SEBlock(input_channels=self.feature_channels)

#         # Classifier layer
#         self.classifier_layer = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),  # Pool to (1,1) size
#             nn.Flatten(),
#             nn.Dropout(0.2),
#             nn.Linear(self.feature_channels, NUM_CLASSES)
#         )

#     def forward(self, x):
#         """
#         Forward pass
#         """
#         _batch_size, _frame1, _c, _h, _w = x.size()
#         x = x.view(_batch_size * _frame1, _c, _h, _w)
#         c_out = self.conv(x)
#         c_out = self.se_block(c_out)  # Apply SE block
#         output = self.classifier_layer(c_out)
#         return output, torch.softmax(output, dim = 1)
