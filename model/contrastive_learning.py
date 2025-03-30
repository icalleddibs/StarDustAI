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
import umap


# STEP 1: Load the data and create triplets that will be used for training
# Triplets: anchor, positive, negative
    # anchor: input of the same class
    # positive: another input  of the same class
    # negative: input from a different class
    
def create_triplets(loader):
    """Create triplets for contrastive training."""
    print("Creating triplets...")
    triplets = []
    class_dict = {}
    
    for batch in loader:
        features, labels = batch
        for i in range(len(features)):
            label = labels[i].item()
            if label not in class_dict:
                class_dict[label] = []
            class_dict[label].append(features[i])
            
    print(class_dict.keys())
    print(class_dict.values())

    for label, instances in class_dict.items():
        other_labels = [l for l in class_dict.keys() if l != label]
        for i in range(len(instances)):
            anchor = instances[i]
            positive = random.choice(instances)
            negative_label = random.choice(other_labels)
            negative = random.choice(class_dict[negative_label])
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
print("Loading Data...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleFluxCNN(NUM_CLASSES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = TripletLoss(margin=1.0)
triplets = create_triplets(train_loader)

# Training Loop
print("Training...")
for epoch in range(10):
    total_loss = 0
    model.train()
    for (anchor, positive, negative) in triplets:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
        anchor_feat = model(anchor.unsqueeze(0), return_embedding=True)
        positive_feat = model(positive.unsqueeze(0), return_embedding=True)
        negative_feat = model(negative.unsqueeze(0), return_embedding=True)
        
        loss = criterion(anchor_feat, positive_feat, negative_feat)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    
# STEP 5: Visualize the embeddings
# This version uses UMAP
# Expect clusters to be similar classes, spread out clusters -> good separation

# Extract Embeddings for Visualization
print("Extracting embeddings for visualization...")
embeddings = []
labels_np = []

model.eval()
with torch.no_grad():
    for batch in train_loader:
        features, class_labels = batch
        features = [f.to(device) for f in features]  # Move to device
        
        embeddings_batch = torch.cat([model(f.unsqueeze(0), return_embedding=True) for f in features], dim=0)
        
        embeddings.append(embeddings_batch.cpu().numpy())
        labels_np.append(class_labels.numpy())

embeddings = np.vstack(embeddings)
labels_np = np.concatenate(labels_np)

# Apply UMAP
print("Applying UMAP...") # it was umap.UMAP(..) and it said umap has no attribute UMAP
reducer = umap(n_neighbors=10, min_dist=0.1, metric='euclidean')
reduced_embeddings = reducer.fit_transform(embeddings)

# Plot Results
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels_np, cmap='jet')
plt.colorbar()
plt.title("UMAP Visualization of Contrastive Embeddings")
plt.show()
