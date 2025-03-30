'''
HOW TO USE:
- define CNN model in cnn_models.py
    - modify the model:
        def forward(self, x, return_embedding=False):
        
    - modify the end of the forward function:
        if return_embedding:
            return out
        # leave the original ending
        out = self.dropout(out) 
        logits = self.fc(out)
        return logits
        
- import the model in this file (below)
'''

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn_experiments.cnn_models import FullFeaturesCNN  # modify this with your model
from cnn_training import train_loader, NUM_CLASSES   # this is for loading data

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# STEP 1: Load the data and create triplets that will be used for training
# Triplets: anchor, positive, negative
    # anchor: input of the same class
    # positive: another input  of the same class
    # negative: input from a different class
    
def create_triplets(loader):
    triplets = []
    for batch in loader:
        features, class_labels = batch
        for i in range(len(features)):
            anchor = features[i]
            label = class_labels[i]
            
            # because we have large dataset, consider doing batching instead of random sampling?
            
            # Positive sample (same class, different instance)
            pos_idx = random.choice([j for j in range(len(features)) if torch.equal(class_labels[j], class_labels[i]) and j != i])
            positive = features[pos_idx]

            # Negative sample (different class)
            neg_idx = random.choice([j for j in range(len(features)) if not torch.equal(class_labels[j], class_labels[i])])
            negative = features[neg_idx]

            triplets.append((anchor, positive, negative))

    return triplets


# STEP 2: Define the contrastive loss function
# Use the triplet loss on the model 
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)  # L2 distance
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()
    


# STEP 3: Define the training loop to optimize on triplet loss
# Use the model you want here
model = FullFeaturesCNN(NUM_CLASSES=3, num_global_features=12)
model.train()

## NOT SURE IF LOADING DATA WILL WORK IF IMPORTING FROM cnn_experiments --------------------------------------------------
triplets = create_triplets(train_loader)

# Optimizer & Loss
# MAY HAVE TO MODIFY HERE BASED ON THE DESIRED PARAMETERS ----------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = TripletLoss(margin=1.0)

# Training loop
for epoch in range(10):
    total_loss = 0
    for (anchor, positive, negative) in triplets:
        anchor_feat = model(anchor.unsqueeze(0), return_embedding=True)
        positive_feat = model(positive.unsqueeze(0), return_embedding=True)
        negative_feat = model(negative.unsqueeze(0), return_embedding=True)

        loss = criterion(anchor_feat, positive_feat, negative_feat)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    

# STEP 4: Determine feature importance
# We will use feature pertubation (how the distances change)
def perturb_feature(data, feature_idx, noise_level=0.1):
    """
    Perturb a specific feature (column) of the input data.
    """
    perturbed_data = data.clone()
    perturbed_data[:, feature_idx] += noise_level * torch.randn_like(perturbed_data[:, feature_idx])
    return perturbed_data

# Loop through the train_loader to get a batch and compute perturbation on features
for batch in train_loader:
    features, class_labels = batch

    # Measure original embeddings
    original_embeddings = model(features, return_embedding=True)

    # Perturb the first feature (feature_idx=0) and compute new embeddings
    perturbed_features = perturb_feature(features, feature_idx=0)
    perturbed_embeddings = model(perturbed_features, return_embedding=True)

    # Calculate the contrastive distance between the original and perturbed embeddings
    distance_change = F.pairwise_distance(original_embeddings, perturbed_embeddings)
    print(f"Feature 0 Change (Perturbation): {distance_change.mean().item()}")

    
    
    
    
# STEP 5: Visualize the embeddings
# Use PCA to visualize the embeddings in 2D space
    # Consider using UMAP due to class material? ------------------------------------------------------------------
# Expect clusters to be similar classes, spread out clusters -> good separation

embeddings = []
labels_np = []

# Have to iterate through train_loader to get embeddings and class labels
for batch in train_loader:
    features, class_labels = batch

    # Forward pass to get embeddings (use return_embedding=True)
    embeddings_batch = model(features, return_embedding=True).detach().numpy()

    # Append the embeddings and labels
    embeddings.append(embeddings_batch)
    labels_np.append(class_labels.numpy())

# Concatenate all embeddings and labels
embeddings = np.vstack(embeddings)
labels_np = np.concatenate(labels_np)

# PCA reduction to 2D for visualization
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Plot the results
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels_np, cmap="jet")
plt.colorbar()
plt.title("Feature Importance in Embedding Space")
plt.show()