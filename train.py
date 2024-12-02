import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split

from model.model import SynthesisPredictionModel
from model.dataset import TrainDataset, train_collate_fn
from model.loss import CustomRankLoss
from evaluate import evaluate_model

# Training loop parameters
num_epochs = 100
log_interval = 1
save_interval = 10
eval_interval = 1

checkpoints_dir = "checkpoints2"
os.makedirs(checkpoints_dir, exist_ok=True)

# Load dataset
dataset = TrainDataset()

# Split dataset into 80% train and 20% evaluate
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=train_collate_fn)
eval_dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=False, collate_fn=train_collate_fn)

model = SynthesisPredictionModel()

# Define optimizer and custom loss function
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = CustomRankLoss(margin=100.0)

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    batch_count = 0

    for batch_idx, (target_formulas, padded_precursor_indexes) in enumerate(train_dataloader):
        # Move data to the same device as the model
        target_formulas = target_formulas.to(device)
        padded_precursor_indexes = [indices.to(device) for indices in padded_precursor_indexes]

        # Forward pass
        logits = model(target_formulas)  # Shape: (batch_size, output_dim)

        # Compute loss
        loss = criterion(logits, padded_precursor_indexes)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    avg_loss = total_loss / batch_count
    print(f"Epoch [{epoch + 1}/{num_epochs}] Complete. Average Loss: {avg_loss:.4f}")

    # Evaluate the model on the eval set every eval_interval epochs
    if (epoch + 1) % eval_interval == 0:
        evaluate_model(model, eval_dataloader, device)

    # Save the model checkpoint every save_interval epochs
    if (epoch + 1) % save_interval == 0:
        checkpoint_path = os.path.join(checkpoints_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")