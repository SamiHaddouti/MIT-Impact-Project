import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.model import SynthesisPredictionModel
from model.loss import CustomRankLoss
from model.loss import mean_reciprocal_rank, top_k_accuracy, contrastive_loss



def train_one_epoch(model: SynthesisPredictionModel,
                    data_loader: DataLoader, 
                    criterion: CustomRankLoss, 
                    optimizer: Adam, 
                    device: torch.device,
                    ) -> float:
    model.train()
    total_loss = 0.0
    batch_count = 0
    if isinstance(criterion, nn.TripletMarginLoss):
        for anchor, positive, negative in data_loader:
            anchor = torch.tensor(anchor).to(device)
            positive = torch.tensor(positive).to(device)
            negative = torch.tensor(negative).to(device)

            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = criterion(anchor_emb, positive_emb, negative_emb)
            print("loss:", loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    if isinstance(criterion, CustomRankLoss):
        for _, (target_formulas, padded_precursor_indexes) in enumerate(data_loader):
            # Move data to the same device as the model
            target_formulas = target_formulas.to(device)
            padded_precursor_indexes = [
                indices.to(device) for indices in padded_precursor_indexes
                ]
            
            logits = model(target_formulas)  # Shape: (batch_size, output_dim)
            
            loss = criterion(logits, padded_precursor_indexes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    else:
        for precursors, targets, labels in data_loader:
            precursors = precursors.to(device)
            targets = targets.to(device)
            labels = labels.to(device)
    
            precursor_embeddings = model(precursors) 
            target_embeddings = model(targets)       

            loss = contrastive_loss(precursor_embeddings, target_embeddings, labels)
            print("loss:", loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1

    avg_loss = total_loss / batch_count
    return avg_loss

def evaluate_model(model: SynthesisPredictionModel, 
                   dataloader: DataLoader, 
                   device: torch.device, 
                   k: int = 10) -> tuple[float, float]:
    model.eval()
    total_mrr = 0.0
    total_top_k = 0.0
    batch_count = 0

    with torch.no_grad():
        for _, (target_formulas, padded_precursor_indexes) in enumerate(dataloader):
            # Move data to the same device as the model
            target_formulas = target_formulas.to(device)
            padded_precursor_indexes = [
                indices.to(device) for indices in padded_precursor_indexes
            ]

            # Forward pass
            logits = model(target_formulas)  # Shape: (batch_size, output_dim)

            # Get the predicted indices sorted by logits (descending order)
            _, predicted_indices = torch.sort(logits, descending=True)

            # Calculate MRR
            batch_mrr = mean_reciprocal_rank(
                predicted_indices, padded_precursor_indexes
            )
            batch_top_k = top_k_accuracy(
                predicted_indices, padded_precursor_indexes, k
            )
            total_mrr += batch_mrr
            total_top_k += batch_top_k
            batch_count += 1

    avg_mrr = total_mrr / batch_count
    avg_top_k = total_top_k / batch_count
    return avg_mrr, avg_top_k