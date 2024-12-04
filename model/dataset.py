import ast
import json 
import random
import pickle
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class CandidateEmbedding():

    def __init__(self, embedding_file: str = "data/formulas_to_embedding.pkl") -> None:
        with open(embedding_file, "rb") as f:
            self.formulas_to_embedding = pickle.load(f)
    
    def get_embedding(self, target_formula: str) -> np.array:
        return np.array(self.formulas_to_embedding[target_formula])


class TrainDataset(Dataset):
    def __init__(self, csv_file: str = "data/ground_truth_sets.csv", use_criterion: str="contrastive", sample_neg_amount: int=3) -> None:
        self.use_criterion = use_criterion
        self.sample_neg_amount = sample_neg_amount
        
        def convert_to_list_of_lists(cell):
            return ast.literal_eval(cell)
        
        self.data = pd.read_csv(csv_file, converters={1: convert_to_list_of_lists})
        self.formulas_to_embedding = CandidateEmbedding().formulas_to_embedding

        with open("data/candidates.json", "r") as f:
            self.candidates = json.load(f)
    
    def __len__(self) -> int:
        return len(self.data)
    
    # def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[int]]:
    #     target_formula = self.data.iloc[idx, 0]
    #     target_formula = self.formulas_to_embedding[target_formula]
    #     precursor_indexes = []
    #     precursor_formulas = self.data.iloc[idx, 1]
    #     for p in precursor_formulas:
    #          precursor_indexes.append(self.precursor_to_index(p))
    #     return target_formula, precursor_indexes

    def __getitem__(self, idx: int) -> Union[Tuple[np.ndarray, List[int]], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self.use_criterion == "triplet":
            negative_idxs = np.random.randint(0, len(self.data), size=(self.sample_neg_amount,))
            return self._get_triplet(idx, negative_idxs=negative_idxs)

        elif self.use_criterion == "contrastive":
            negative_idxs = np.random.randint(0, len(self.data), size=(self.sample_neg_amount,))
            return self._get_contrastive(idx, negative_idxs=negative_idxs)

        else:
            return self._get_default(idx)

    def _get_triplet(self, idx: int, negative_idxs: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_embd = self._get_target(idx) #anchor
        positive_embd = self._get_positive(idx) 
        negative_embd = random.choice(self._get_negatives(negative_idxs))  
        return target_embd, positive_embd, negative_embd

    def _get_contrastive(self, idx: int, negative_idxs: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        target_embd = self._get_target(idx)
        positive_embd = self._get_positive(idx)
        # negative_embd = random.choice(self._get_negatives(negative_idxs))  
        # return positive_pair, negative_pairs
        return positive_embd, target_embd 

    def _get_default(self, idx: int) -> Tuple[np.ndarray, List[int]]:
        target_formula = self.data.iloc[idx, 0]
        target_formula = self.formulas_to_embedding[target_formula]
        precursor_indexes = []
        precursor_formulas = self.data.iloc[idx, 1]
        for p in precursor_formulas:
            precursor_indexes.append(self.precursor_to_index(p))
        return target_formula, precursor_indexes
    
    def _get_target(self, idx: int) -> torch.Tensor:
        target_formula = self.data.iloc[idx, 0]
        target_embd = torch.from_numpy(self.formulas_to_embedding[target_formula]).float()
        return target_embd
    
    def _get_positive(self, idx: int) -> torch.Tensor:
        precursor_formulas = self.data.iloc[idx, 1]
        for p in precursor_formulas:
            temp_prec = np.array([self.formulas_to_embedding[subp] for subp in p])
            prec_embd = torch.from_numpy(np.mean(temp_prec, axis=0).astype(np.float32))
        return prec_embd

    def _get_negatives(self, negative_idxs: List[int]) -> List[torch.Tensor]:
        negatives = []
        for neg_idx in negative_idxs:
            negative_formula = self.data.iloc[neg_idx, 0]
            neg_embd = torch.from_numpy(self.formulas_to_embedding[negative_formula]).float()
            negatives.append(neg_embd)
        return negatives

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

def train_collate_fn(batch: List[Tuple[np.ndarray, List[int]]], criterion: str="contrastive") -> Tuple[torch.Tensor, torch.Tensor]:

    if criterion == "contrastive":
        precursors = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        #positive pairs
        positive_pairs = list(zip(precursors, targets))
        labels = [1] * len(positive_pairs)

        #negative pairs
        negative_pairs = []
        for i, target in enumerate(targets):
            while True:
                negative_idx = int(np.random.randint(0, len(targets), size=(1,)))
                if negative_idx != i: break
            negative_sample = precursors[negative_idx]
            negative_pair = (target, negative_sample)
            negative_pairs.append(negative_pair)
        
        pairs = positive_pairs + negative_pairs
        labels += [0] * len(negative_pairs)
        
        combined = list(zip(pairs, labels))
        random.shuffle(combined)

        pairs, labels = zip(*combined)
        precursors, targets = zip(*pairs)
        
        precursors = torch.stack([torch.tensor(prec) for prec in precursors])  
        targets = torch.stack([torch.tensor(target) for target in targets])  
        labels = torch.tensor(labels).view(-1,1)
        return precursors, targets, labels

    elif criterion == "triplet":
        anchors = []
        positives = []
        negatives = []

        for item in batch:
            anchor, positive, negative = item
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)
    
    else:    
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