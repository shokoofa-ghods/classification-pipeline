"""
Module containing Video model
"""

import torch
import torch.nn as nn
import torchvision

from common.constant import (INITIAL_LABELS,
                            COMBINED_VIEW_MAP,
                            VIEW_MAP)

CNN_OUTPUT_SIZE = 1000
NUM_CLASSES = len(VIEW_MAP)
HIDDEN_SIZE = 128


# CNN Backbone (e.g., EfficientNet-b0 or Resnet18) + Temporal Model (Transformer Module)

class TransformerBased_video(nn.Module):
    def __init__(self, model_type, hidden_dim=256, num_layers=2, nhead=4, seq_len=10):
        super(TransformerBased_video, self).__init__()

        self.group_names = VIEW_MAP #['plax', 'psax-ves', 'psax-sub', 'apical-2ch', 'apical-3ch', 'apical-4&5ch', 'suprasternal', 'subcostal']
        self.view_id_to_group_idx = {
            view_id: self.group_names.index(COMBINED_VIEW_MAP[view_id])
            for view_id in range(len(INITIAL_LABELS))
        }

        self.backbone_type = model_type['backbone']
        self.feature_channels = model_type['feature_channels']

        if self.backbone_type.startswith('resnet18'):
            base_model = torchvision.models.resnet18(weights='DEFAULT')
            self.features = nn.Sequential(*list(base_model.children())[:-2])

        elif self.backbone_type.startswith('efficientnet-b0'):
            backbone = torchvision.models.efficientnet_b0(weights='DEFAULT')
            self.backbone = backbone.features

        else: # EfficientNet-B2 is Default
            backbone = torchvision.models.efficientnet_b2(weights='DEFAULT')
            self.backbone = backbone.features

            
        self.dropout2d = nn.Dropout2d(p=0.2)
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten(start_dim=2)

        # Linear projection to transformer hidden dim
        self.linear_proj = nn.Linear(self.feature_channels * 4 * 4, hidden_dim)

        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.temporal_type = self.transformer.__class__.__name__

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, NUM_CLASSES)
        )

    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.backbone(x)
        x = self.spatial_pool(x)
        x = self.flatten(x).view(B, T, -1)
        x = self.linear_proj(x)  # [B, T, hidden_dim]

        x = x + self.pos_embedding[:, :T, :]  # Add positional encodings

        x = self.transformer(x)  # [B, T, hidden_dim]

        # Attention pooling
        attn_weights = torch.softmax(self.attention(x), dim=1)  # [B, T, 1]
        x = torch.sum(x * attn_weights, dim=1)  # [B, hidden_dim]

        return self.classifier(x)
 

# ---------------------------------------------------------------------------------------------------------------------------------

# CNN Backbone (e.g., EfficientNet-b2 or ResNet 18) + Temporal model (Bidirectional GRU) + Attention Module for frame info fusion
class Spatial_Temporal(nn.Module):
    def __init__(self, model_type ):
        super(Spatial_Temporal, self).__init__()

        self.backbone_type = model_type['backbone']
        self.feature_channels = model_type['feature_channels']

        self.group_names = VIEW_MAP # ['plax', 'psax-ves', 'psax-sub', 'apical-2ch', 'apical-3ch', 'apical-4&5ch', 'suprasternal', 'subcostal']
        self.view_id_to_group_idx = {
            view_id: self.group_names.index(COMBINED_VIEW_MAP[view_id])
            for view_id in range(len(INITIAL_LABELS))
        }


        if self.backbone_type.startswith('resnet18'):
            base_model = torchvision.models.resnet18(weights='DEFAULT')
            self.features = nn.Sequential(*list(base_model.children())[:-2])

        elif self.backbone_type.startswith('efficientnet-b0'):
            backbone = torchvision.models.efficientnet_b0(weights='DEFAULT')
            self.backbone = backbone.features

        else: # EfficientNet-B2 is Default
            backbone = torchvision.models.efficientnet_b2(weights='DEFAULT')
            self.backbone = backbone.features
            
        self.dropout2d = nn.Dropout2d(p=0.2)
        self.spatial_pool = nn.AdaptiveAvgPool2d((4,4))
        self.flatten = nn.Flatten(start_dim=2) # flatten H and W

        self.rnn = nn.GRU( self.feature_channels * 4 * 4, HIDDEN_SIZE, batch_first=True, bidirectional=True )
        self.temporal_type = self.rnn.__class__.__name__

        self.temporal_attention = nn.Sequential( 
            nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE//2 ), # Due to Bidirectional GRU the ouput is Double of Hidden Size
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE//2, 1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_SIZE * 2, NUM_CLASSES)
        )

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        x = self.features(x) # [B*T, C, H', W']
        x = self.spatial_pool(x) # [B*T, C, 4, 4]
        x = self.dropout2d(x)
        x = self.flatten(x).view(B, T, -1) # [B, T, C*H*W]

        rnn_out, _ = self.rnn(x) # [B, T, 2*hidden]
        attn_weights = torch.softmax(self.temporal_attention(rnn_out), dim = 1) # [B, T, 1]
        x = torch.sum(rnn_out * attn_weights, dim = 1) # [B, hidden]
        
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------------------------------------------------------------

# CNN Backbone (EficientNet-b2 default) + Temporal Model (LSTM Module) 
class CNNLSTM(nn.Module):
    """
    Video CNNLSTM Model
    """
    def __init__(self, model_type, num_layers:int=2):
        super(CNNLSTM, self).__init__()
        self.num_layers = num_layers

        self.backbone_type = model_type['backbone']
        self.feature_channels = model_type['feature_channels']
        self.temporal_type = model_type['temporal']

        
        if self.backbone_type.startswith('resnet18'):
            base_model = torchvision.models.resnet18(weights='DEFAULT')
            self.features = nn.Sequential(*list(base_model.children())[:-2])

        elif self.backbone_type.startswith('efficientnet-b0'):
            backbone = torchvision.models.efficientnet_b0(weights='DEFAULT')
            self.backbone = backbone.features

        else: # EfficientNet-B2 is Default
            backbone = torchvision.models.efficientnet_b2(weights='DEFAULT')
            self.backbone = backbone.features
            # for module in self.conv.features:
            #     if isinstance(module, torch.nn.modules.container.Sequential):
            #         module.append(torch.nn.Dropout(0.4))

        self.lstm = nn.LSTM(self.feature_channels,
                            HIDDEN_SIZE,
                            self.num_layers,
                            batch_first=True,
                            dropout = 0.1)
        
        self.temporal_type = self.lstm.__class__.__name__

        self.attention_layer = nn.Linear(HIDDEN_SIZE, 1)
        self.classifier_layer = nn.Sequential(nn.Dropout(0.2),
                                              nn.Linear(HIDDEN_SIZE, NUM_CLASSES))

    def attention_net(self, lstm_out:torch.Tensor) -> torch.Tensor:
        """
        Apply the attention layer to the LSTM output

        Args:
            lstm_out (Tensor): output of the lstm block
        
        Returns:
            The Tensor output of the attention layer
        """
        attention_weights = torch.tanh(self.attention_layer(lstm_out))
        attention_weights = torch.softmax(attention_weights, dim=1)
        attention_out = torch.bmm(attention_weights.transpose(1, 2), lstm_out)
        return attention_out.squeeze(1)

    def forward(self, x):
        """
        Forward pass
        """
        batch_size, seq_len, c, h, w = x.size()
        c_in = x.view(batch_size * seq_len, c, h, w)
        # print(self.conv)
        c_out = self.backbone(c_in)
        lstm_in = c_out.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(lstm_in)

        lstm_out = lstm_out[:, -1, :]
        output = self.classifier_layer(lstm_out)

        # attention_out = self.attention_net(lstm_out)
        # output = self.classifier_layer(attention_out)

        return output, torch.softmax(output, dim = 1)
