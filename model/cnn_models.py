# Deep Learning
import torch.nn as nn
import torch.nn.functional as F
import torch


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

        out = F.relu(self.conv1(flux))
        out = self.ln1(out)  
        out = self.dropout(out)

        out = F.relu(self.conv2(out))
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

        out = F.relu(self.conv1(flux))
        out = self.ln1(out)
        out = self.dropout(out)

        out = F.relu(self.conv2(out))
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

        out = F.relu(self.conv1(combined_input))
        out = self.ln1(out)
        out = self.dropout(out)

        out = F.relu(self.conv2(out))
        out = self.ln2(out)
        out = self.dropout(out)

        out = self.pool(out).squeeze(2)

        out = self.dropout(out)  # Dropout before FC
        logits = self.fc(out)
        return logits