# Deep Learning
import torch.nn as nn
import torch.nn.functional as F
import torch


class FullFeaturesResNet(nn.Module):
    """
    CNN model with residual connections for flux-based classification.
    """
    def __init__(self, NUM_CLASSES=3, num_global_features=12, dropout_rate=0.3):
        super(FullFeaturesResNet, self).__init__()
        self.initial_conv = nn.Conv1d(in_channels=1 + num_global_features, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn_initial = nn.BatchNorm1d(16)

        self.res_block1 = ResidualBlock(16, 32, dropout_rate)
        self.res_block2 = ResidualBlock(32, 64, dropout_rate)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, NUM_CLASSES)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)."""
        flux = x[:, :, 0]  
        global_features = x[:, 0, 2:]

        # Expand global features to match flux length
        global_features = global_features.unsqueeze(2).expand(-1, -1, flux.size(1))

        flux = flux.unsqueeze(1)
        combined_input = torch.cat((flux, global_features), dim=1)

        out = F.leaky_relu(self.bn_initial(self.initial_conv(combined_input)))

        out = self.res_block1(out)
        out = self.res_block2(out)

        out = self.pool(out).squeeze(2)
        out = self.dropout(out)  
        logits = self.fc(out)
        return logits


##### Other helper classes #####

class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    """
    def __init__(self, patience=5, verbose=0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        """
        Args:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): Model being trained.
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0



class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for multi-class classification.
        Args:
            alpha (list or None): Class weights for focal loss. If None, defaults to [0.16, 0.31, 0.53].
            gamma (float): Focusing parameter.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is None:
            self.alpha = torch.tensor([0.16, 0.31, 0.53])  # Match class imbalance
        else:
            self.alpha = torch.tensor(alpha)
    
    def forward(self, inputs, targets):
        """
        Forward pass for focal loss.
        Args:
            inputs (torch.Tensor): Model predictions (logits).
            targets (torch.Tensor): True labels.
        Returns:
            torch.Tensor: Computed focal loss.
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(-1)).float()
        
        # Calculate focal weight
        focal_weight = (1 - probs) ** self.gamma
        loss = -self.alpha.to(inputs.device) * focal_weight * log_probs * targets_one_hot
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.3):
        """
        Residual block with two convolutional layers and a skip connection.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dropout_rate (float): Dropout rate.
        """ 
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=dropout_rate)

        # 1x1 conv to match dimensions if needed
        self.residual_projection = None
        if in_channels != out_channels:
            self.residual_projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the residual block.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after residual connection.
        """
        identity = x  # Save input for residual connection

        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)

        if self.residual_projection is not None:
            identity = self.residual_projection(identity)

        out += identity  # Residual connection
        return F.leaky_relu(out)

