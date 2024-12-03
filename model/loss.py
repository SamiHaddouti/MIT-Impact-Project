from typing import List

import torch
import torch.nn as nn

class CustomRankLoss(nn.Module):
    def __init__(self, margin: float=10.0):
        """
        Custom loss to ensure the highest logits correspond to the correct indices.
        Args:
            margin (float): Minimum margin by which correct logits must exceed incorrect logits.
        """
        super(CustomRankLoss, self).__init__()
        self.margin = margin

    def forward(self, logits: torch.Tensor, padded_correct_indices: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Model output of shape (batch_size, num_classes).
            correct_indices (List[torch.Tensor]): List of tensors where each tensor contains the correct indices for each example in the batch.
        Returns:
            torch.Tensor: Computed loss.
        """
        batch_size = logits.size(0)
        loss = 0.0

        for i in range(batch_size):
            correct_indices = padded_correct_indices[i]
            valid_indices = correct_indices[correct_indices >= 0]
            correct_logits = logits[i, valid_indices]  # Logits at correct indices

            # Get the logits for all other indices (incorrect logits)
            incorrect_logits = logits[i]
            incorrect_logits = incorrect_logits[
                torch.isin(
                    elements=torch.arange(logits.size(1), device=logits.device), 
                    test_elements=valid_indices, 
                    invert=True,
                    )
                ]  # Remove correct indices
            
            # Margin-based ranking loss
            # Make sure to unsqueeze dimensions for broadcasting (correct_logits: (num_correct, 1), incorrect_logits: (num_incorrect,))
            pairwise_losses = torch.relu(self.margin + incorrect_logits.unsqueeze(0) - correct_logits.unsqueeze(1))
            
            # Mean pairwise loss for this example
            loss += pairwise_losses.mean()

        return loss / batch_size


def mean_reciprocal_rank(predicted_indices: torch.Tensor, 
                         correct_indices: torch.Tensor) -> float:
    """
    Calculate the Mean Reciprocal Rank (MRR) for the batch.
    
    Args:
        predicted_indices (torch.Tensor): Tensor of shape (batch_size, num_classes) 
                                          with the predicted ranks.
        correct_indices (torch.Tensor): Tensor of shape (batch_size, num_correct) 
                                        with the correct indices.
    
    Returns:
        float: Mean Reciprocal Rank for the batch.
    """
    batch_size = predicted_indices.size(0)
    reciprocal_ranks = []
    
    for i in range(batch_size):
        correct_index_set = set(
            idx for idx in correct_indices[i].tolist() if idx >= 0
            )
        for rank, idx in enumerate(predicted_indices[i].tolist(), start=1):
            if idx in correct_index_set:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return sum(reciprocal_ranks) / batch_size


def top_k_accuracy(predicted_indices: torch.Tensor, 
                   correct_indices: torch.Tensor, 
                   k: int) -> float:
    """
    Calculate the Top-k accuracy for the batch.
    
    Args:
        predicted_indices (torch.Tensor): Tensor of shape (batch_size, num_classes) 
                                          with the predicted ranks.
        correct_indices (torch.Tensor): Tensor of shape (batch_size, num_correct) 
                                        with the correct indices.
        k (int): The top-k value for calculating Top-k accuracy.
    
    Returns:
        float: Top-k Accuracy for the batch.
    """
    batch_size = predicted_indices.size(0)
    top_k_hits = 0
    
    for i in range(batch_size):
        correct_index_set = set(
            idx for idx in correct_indices[i].tolist() if idx >= 0
            )
        top_k_predictions = set(predicted_indices[i, :k].tolist())
        
        if correct_index_set.intersection(top_k_predictions):
            top_k_hits += 1
    
    return top_k_hits / batch_size