# Deep Learning
import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleFluxCNN(nn.Module):
    """
    Simple CNN model for flux-based classification.
    """

    def __init__(self, NUM_CLASSES=3, dropout_rate=0.3):
        super(SimpleFluxCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.ln1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.ln2 = nn.BatchNorm1d(32)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(32, NUM_CLASSES)

    def forward(self, x):
        flux = x[:, :, 0]  # shape: (batch_size, max_rows)
        flux = flux.unsqueeze(1)

        out = F.leaky_relu(self.conv1(flux))
        out = self.ln1(out)  
        out = self.dropout(out)

        out = F.leaky_relu(self.conv2(out))
        out = self.ln2(out)  
        out = self.dropout(out)

        out = self.pool(out).squeeze(2)

        out = self.dropout(out) 
        logits = self.fc(out)
        return logits


class AllFeaturesCNN(nn.Module):
    """
    CNN model for flux-based classification with concatenated global tags.
    """

    def __init__(self, NUM_CLASSES=3, num_global_features=12, dropout_rate=0.3):
        super(AllFeaturesCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.ln1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.ln2 = nn.BatchNorm1d(32)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(32 + num_global_features, NUM_CLASSES)

    def forward(self, x):
        flux = x[:, :, 0]  # shape: (batch_size, max_rows)
        flux = flux.unsqueeze(1)

        out = F.leaky_relu(self.conv1(flux))
        out = self.ln1(out)
        out = self.dropout(out)

        out = F.leaky_relu(self.conv2(out))
        out = self.ln2(out)
        out = self.dropout(out)

        out = self.pool(out).squeeze(2)

        # Get global features (skip loglam)
        global_features = x[:, 0, 2:]  
        combined = torch.cat((out, global_features), dim=1)

        combined = self.dropout(combined)  # Dropout before FC
        logits = self.fc(combined)
        return logits


class FullFeaturesCNN(nn.Module):
    """
    CNN model for flux-based classification with all features used as CNN input.
    """

    def __init__(self, NUM_CLASSES=3, num_global_features=12, dropout_rate=0.3):
        super(FullFeaturesCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1 + num_global_features, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.ln1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.ln2 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(32, NUM_CLASSES)

    def forward(self, x):
        flux = x[:, :, 0]  # shape: (batch_size, max_rows)
        global_features = x[:, 0, 2:]

        # Expand global features to match flux length
        global_features = global_features.unsqueeze(2).expand(-1, -1, flux.size(1))

        flux = flux.unsqueeze(1)
        combined_input = torch.cat((flux, global_features), dim=1)

        out = F.leaky_relu(self.conv1(combined_input))
        out = self.ln1(out)
        out = self.dropout(out)

        out = F.leaky_relu(self.conv2(out))
        out = self.ln2(out)
        out = self.dropout(out)

        out = self.pool(out).squeeze(2)

        out = self.dropout(out)  # Dropout before FC
        logits = self.fc(out)
        return logits
    
class DilatedFullFeaturesCNN(nn.Module):
    """
    CNN model for flux-based classification with dilated convolutions.
    """

    def __init__(self, NUM_CLASSES=3, num_global_features=12, dropout_rate=0.3, dilation=2):
        super(DilatedFullFeaturesCNN, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1 + num_global_features, 
            out_channels=16, 
            kernel_size=3, 
            stride=1, 
            padding=dilation,  
            dilation=dilation
        )
        self.ln1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(
            in_channels=16, 
            out_channels=32, 
            kernel_size=3, 
            stride=1, 
            padding=dilation, 
            dilation=dilation
        )
        self.ln2 = nn.BatchNorm1d(32)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(32, NUM_CLASSES)

    def forward(self, x):
        flux = x[:, :, 0]  # shape: (batch_size, max_rows)
        global_features = x[:, 0, 2:]

        # Expand global features to match flux length
        global_features = global_features.unsqueeze(2).expand(-1, -1, flux.size(1))

        flux = flux.unsqueeze(1)
        combined_input = torch.cat((flux, global_features), dim=1)

        # First dilated convolution
        out = F.relu(self.conv1(combined_input))
        out = self.ln1(out)
        out = self.dropout(out)

        # Second dilated convolution
        out = F.relu(self.conv2(out))
        out = self.ln2(out)
        out = self.dropout(out)

        # Global average pooling
        out = self.pool(out).squeeze(2)

        out = self.dropout(out)  # Dropout before FC
        logits = self.fc(out)
        return logits

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
        flux = x[:, :, 0]  # shape: (batch_size, max_rows)
        global_features = x[:, 0, 2:]

        # Expand global features to match flux length
        global_features = global_features.unsqueeze(2).expand(-1, -1, flux.size(1))

        flux = flux.unsqueeze(1)
        combined_input = torch.cat((flux, global_features), dim=1)

        out = F.leaky_relu(self.bn_initial(self.initial_conv(combined_input)))

        out = self.res_block1(out)
        out = self.res_block2(out)

        out = self.pool(out).squeeze(2)
        out = self.dropout(out)  # Dropout before FC
        logits = self.fc(out)
        return logits



class FullFeaturesCNNMoreLayers(nn.Module):
    """
    CNN model for flux-based classification with all features used as CNN input.
    """
    def __init__(self, NUM_CLASSES=3, num_global_features=12, dropout_rate=0.3):
        super(FullFeaturesCNNMoreLayers, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1 + num_global_features, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.ln1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.ln2 = nn.BatchNorm1d(32)
        
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.ln3 = nn.BatchNorm1d(64)
        
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.ln4 = nn.BatchNorm1d(128)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        flux = x[:, :, 0]  # shape: (batch_size, max_rows)
        global_features = x[:, 0, 2:]

        # Expand global features to match flux length
        global_features = global_features.unsqueeze(2).expand(-1, -1, flux.size(1))

        flux = flux.unsqueeze(1)
        combined_input = torch.cat((flux, global_features), dim=1)

        out = F.relu(self.conv1(combined_input))
        out = self.ln1(out)
        out = self.dropout(out)

        out = F.relu(self.conv2(out))
        out = self.ln2(out)
        out = self.dropout(out)
        
        out = F.relu(self.conv3(out))
        out = self.ln3(out)
        out = self.dropout(out)
        
        out = F.relu(self.conv4(out))
        out = self.ln4(out)
        out = self.dropout(out)

        out = self.pool(out).squeeze(2)
        out = self.dropout(out)  # Dropout before FC
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
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is None:
            self.alpha = torch.tensor([0.16, 0.31, 0.53])  # Match class imbalance
        else:
            self.alpha = torch.tensor(alpha)
    
    def forward(self, inputs, targets):
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
        identity = x  # Save input for residual connection

        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)

        # Adjust dimensions if necessary
        if self.residual_projection is not None:
            identity = self.residual_projection(identity)

        out += 0.5 *  identity  # Residual connection
        return F.leaky_relu(out)

