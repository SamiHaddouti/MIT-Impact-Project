import torch

from model.loss import mean_reciprocal_rank


def evaluate_model(model, dataloader, device):
    model.eval()
    total_mrr = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch_idx, (target_formulas, padded_precursor_indexes) in enumerate(dataloader):
            # Move data to the same device as the model
            target_formulas = target_formulas.to(device)
            padded_precursor_indexes = [indices.to(device) for indices in padded_precursor_indexes]

            # Forward pass
            logits = model(target_formulas)  # Shape: (batch_size, output_dim)

            # Get the predicted indices sorted by logits (descending order)
            _, predicted_indices = torch.sort(logits, descending=True)

            # Calculate MRR
            batch_mrr = mean_reciprocal_rank(predicted_indices, torch.stack(padded_precursor_indexes))
            total_mrr += batch_mrr
            batch_count += 1

    avg_mrr = total_mrr / batch_count
    print(f"Evaluation Complete. Average MRR: {avg_mrr:.4f}")
    return avg_mrr