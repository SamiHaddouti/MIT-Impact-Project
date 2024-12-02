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

    def forward(self, logits, padded_correct_indices):
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
            
            # print(f"Length of output vector: {len(logits[i])}")
            # print(f"Length of correct indices: {len(correct_indices)}")
            # print(correct_indices)
            # print(f"Length of valid indices: {len(valid_indices)}")
            # print(valid_indices)
            # print(f"Length of correct logits: {len(correct_logits)}")
            # print(correct_logits)
            # print(f"Length of incorrect logits: {len(incorrect_logits)}")
            # print("")
            
            # Margin-based ranking loss
            # Make sure to unsqueeze dimensions for broadcasting (correct_logits: (num_correct, 1), incorrect_logits: (num_incorrect,))
            pairwise_losses = torch.relu(self.margin + incorrect_logits.unsqueeze(0) - correct_logits.unsqueeze(1))
            
            # Mean pairwise loss for this example
            loss += pairwise_losses.mean()

        return loss / batch_size


def mean_reciprocal_rank_and_top_k(predicted_indices, correct_indices, k):
    """
    Calculate the Mean Reciprocal Rank (MRR) and Top-k accuracy for the batch.
    
    Args:
        predicted_indices (torch.Tensor): Tensor of shape (batch_size, num_classes) with the predicted ranks.
        correct_indices (torch.Tensor): Tensor of shape (batch_size, num_correct) with the correct indices.
        k (int): The top-k value for calculating Top-k accuracy.
    
    Returns:
        tuple: Mean Reciprocal Rank (float), Top-k Accuracy (float) for the batch.
    """
    batch_size = predicted_indices.size(0)
    reciprocal_ranks = []
    top_k_hits = 0
    
    for i in range(batch_size):
        # Get the correct indices as a set
        correct_index_set = set(correct_indices[i].tolist())
        
        # Calculate Reciprocal Rank
        for rank, idx in enumerate(predicted_indices[i].tolist(), start=1):
            if idx in correct_index_set:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)  # No correct index found in the predictions
        
        # Calculate Top-k Accuracy
        top_k_predictions = set(predicted_indices[i, :k].tolist())
        if correct_index_set.intersection(top_k_predictions):
            top_k_hits += 1
    
    mrr = sum(reciprocal_ranks) / batch_size
    top_k_accuracy = top_k_hits / batch_size
    
    return mrr, top_k_accuracy