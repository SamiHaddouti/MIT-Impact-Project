import ast
import json 
import numpy as np
import pandas as pd
from typing import List, Tuple
import pickle
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class CandidateEmbedding():

    def __init__(self, embedding_file: str = "data/formulas_to_embedding.pkl") -> None:
        with open(embedding_file, "rb") as f:
            self.formulas_to_embedding = pickle.load(f)
    
    def get_embedding(self, target_formula: str) -> np.array:
        return np.array(self.formulas_to_embedding[target_formula])


class TrainDataset(Dataset):
    def __init__(self, csv_file: str = "data/ground_truth_sets.csv") -> None:

        def convert_to_list_of_lists(cell):
            return ast.literal_eval(cell)
        
        self.data = pd.read_csv(csv_file, converters={1: convert_to_list_of_lists})

        self.formulas_to_embedding = CandidateEmbedding().formulas_to_embedding

        with open("data/candidates.json", "r") as f:
            self.candidates = json.load(f)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[int]]:
        target_formula = self.data.iloc[idx, 0]
        target_formula = self.formulas_to_embedding[target_formula]
        precursor_indexes = []
        precursor_formulas = self.data.iloc[idx, 1]
        for p in precursor_formulas:
             precursor_indexes.append(self.precursor_to_index(p))
        return target_formula, precursor_indexes

    def precursor_to_index(self, precursor_set: List[str]) -> int:
        try:
            index = self.candidates.index(precursor_set)
        except ValueError:
            index = -1  # Return -1 if the precursor_set is not found
        return index
    
    def get_meanpooled_candidate_embeddings(self):
        print("Calculating mean pooled candidate embeddings")
        meanpooled_candidate_embeddings = []
        for candidate in self.candidates:
            candidate_embedding = np.mean(
                [self.formulas_to_embedding[formula] for formula in candidate],
                axis=0
            )
            meanpooled_candidate_embeddings.append(candidate_embedding)
        meanpooled_candidate_embeddings = np.array(meanpooled_candidate_embeddings)
        return torch.tensor(meanpooled_candidate_embeddings, dtype=torch.float32)

def train_collate_fn(batch: List[Tuple[np.ndarray, List[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
    target_formulas, precursor_indexes = zip(*batch)
    
    # Convert target formulas to tensors
    target_formulas = [torch.tensor(tf, dtype=torch.float32) for tf in target_formulas]
    
    # Pad precursor indexes to the same length
    max_length = max(len(pf) for pf in precursor_indexes)
    padded_precursor_indexes = [
        torch.tensor(pf + [-1] * (max_length - len(pf)), dtype=torch.long) for pf in precursor_indexes
    ]
    
    return torch.stack(target_formulas), torch.stack(padded_precursor_indexes)


class TestDataset(Dataset):
            def __init__(self, json_file: str = "data/test_targets.json") -> None:
                with open(json_file, 'r') as f:
                    self.data = json.load(f)
                self.formulas_to_embedding = CandidateEmbedding().formulas_to_embedding
            
            def __len__(self) -> int:
                return len(self.data)
            
            def __getitem__(self, idx: int) -> np.ndarray:
                target_formula = self.data[idx]
                target_formula = self.formulas_to_embedding[target_formula]
                return target_formula