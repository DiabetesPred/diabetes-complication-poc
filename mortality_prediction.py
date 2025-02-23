import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyhealth.datasets import MIMIC3Dataset
import logging
from collections import Counter
from tqdm import tqdm
from multiprocessing import freeze_support
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
import random
import copy
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def mortality_prediction_mimic3_fn(patient):
    """
    Mortality prediction aims at predicting whether the patient will decease in the
    next hospital visit based on the clinical information from current visit.
    """
    samples = []
    for visit_idx in range(len(patient) - 1):
        visit = patient[visit_idx]
        next_visit = patient[visit_idx + 1]

        # obtain the label
        mortality_label = int(next_visit.discharge_status == 1)  # 1 for deceased, 0 for others

        # step 1: obtain features
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")

        # step 2: exclusion criteria
        if len(conditions) + len(procedures) == 0: 
            continue

        # step 3: assemble the sample
        samples.append({
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,
            "conditions": conditions,
            "procedures": procedures,
            "label": mortality_label,
        })
    
    return samples

def new_mortality_prediction_mimic3_fn(patient):
    """
    Mortality prediction aims at predicting whether the patient will decease in this
        hospital visit based on the clinical information from current visit
        (e.g., conditions and procedures).

    """
    samples = []
    for visit in patient:

        # obtain the label
        mortality_label = int(visit.discharge_status)

        # step 1: obtain features
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")

        # step 2: exclusion criteria
        if len(conditions) + len(procedures) == 0: continue

        # step 3: assemble the sample
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "label": mortality_label,
            }
        )
    
    return samples

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
            self.counter = 0

class MortalityDataset(Dataset):
    def __init__(self, samples, vocab_size, augment=False):
        self.samples = samples
        self.vocab_size = vocab_size
        self.augment = augment
        
        # Calculate class weights for sampling
        self.labels = [sample['label'] for sample in samples]
        label_counts = Counter(self.labels)
        self.weights = [1.0 / label_counts[label] for label in self.labels]
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        features = torch.zeros(self.vocab_size)
        codes = sample['conditions'] + sample['procedures']
        
        for code in codes:
            code_idx = hash(str(code)) % self.vocab_size
            features[code_idx] = 1
            
        if self.augment and len(codes) > 2:
            # More sophisticated augmentation
            if random.random() < 0.5:
                # Randomly drop 10-20% of the codes
                drop_rate = random.uniform(0.1, 0.2)
                mask = torch.rand(len(codes)) > drop_rate
                for i, (include, code) in enumerate(zip(mask, codes)):
                    if not include:
                        code_idx = hash(str(code)) % self.vocab_size
                        features[code_idx] = 0
            else:
                # Randomly duplicate 10-20% of the codes
                dup_rate = random.uniform(0.1, 0.2)
                mask = torch.rand(len(codes)) < dup_rate
                for i, (duplicate, code) in enumerate(zip(mask, codes)):
                    if duplicate:
                        code_idx = hash(str(code)) % self.vocab_size
                        features[code_idx] = 1
            
        return features, torch.tensor(sample['label'], dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class MortalityPredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate=0.5, use_batch_norm=True):
        super().__init__()
        
        # Layer normalization instead of batch normalization for input
        self.input_norm = nn.LayerNorm(input_size)
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers with residual connections
        for hidden_size in hidden_sizes:
            block = []
            block.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                # Use Layer Normalization instead of Batch Normalization
                block.append(nn.LayerNorm(hidden_size))
            block.append(nn.ReLU())
            block.append(nn.Dropout(dropout_rate))
            
            # Add residual connection if sizes match
            if prev_size == hidden_size:
                layers.append(ResidualBlock(nn.Sequential(*block)))
            else:
                layers.extend(block)
            
            prev_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, 1)
        
    def forward(self, x):
        x = self.input_norm(x)
        x = self.hidden_layers(x)
        return self.output_layer(x).view(-1)

class ResidualBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        
    def forward(self, x):
        return x + self.block(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=20):
    early_stopping = EarlyStopping(patience=5, verbose=True)
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        for features, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            outputs = outputs.view(-1)
            labels = labels.float().view(-1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = total_train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_batches = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                outputs = outputs.view(-1)
                labels = labels.float().view(-1)
                
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                val_batches += 1
                
                val_preds.extend((torch.sigmoid(outputs) > 0.5).cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        val_accuracy = accuracy_score(val_true, val_preds)
        val_accuracies.append(val_accuracy)
        
        logger.info(f'Epoch {epoch+1}/{epochs}:')
        logger.info(f'Train Loss: {avg_train_loss:.4f}')
        logger.info(f'Val Loss: {avg_val_loss:.4f}')
        logger.info(f'Val Accuracy: {val_accuracy:.4f}')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            model.load_state_dict(early_stopping.best_model.state_dict())
            break
    
    return train_losses, val_losses, val_accuracies

def evaluate_model(model, test_loader, device):
    model.eval()
    test_preds = []
    test_true = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs.view(-1))
            test_preds.extend((probs > 0.5).cpu().numpy())
            test_true.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays and ensure integer type
    test_true = np.array(test_true, dtype=int)
    test_preds = np.array(test_preds, dtype=int)
    
    # Calculate metrics safely
    metrics = {
        'accuracy': accuracy_score(test_true, test_preds),
        'confusion_matrix': confusion_matrix(test_true, test_preds)
    }
    
    # Calculate class distribution
    unique, counts = np.unique(test_true, return_counts=True)
    metrics['class_distribution'] = dict(zip(unique, counts))
    
    # Only calculate other metrics if we have both classes
    if len(unique) > 1:
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_true, 
            test_preds, 
            average='binary',
            zero_division=0
        )
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': roc_auc_score(test_true, test_preds)
        })
    else:
        metrics.update({
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc_roc': 0.0
        })
        logger.warning("Only one class present in the test set. Some metrics are not meaningful.")
    
    return metrics

def plot_training_history(train_losses, val_losses, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    logger.info("Training history plot saved as 'training_history.png'")

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    logger.info("Confusion matrix plot saved as 'confusion_matrix.png'")

def train_with_kfold(dataset, vocab_size, device, hyperparameters, n_splits=5, batch_size=8):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_models = []
    best_val_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset.samples)):
        logger.info(f"\nTraining Fold {fold + 1}/{n_splits}")
        
        # Split data
        train_samples = [dataset.samples[i] for i in train_idx]
        val_samples = [dataset.samples[i] for i in val_idx]
        
        # Create datasets
        train_dataset = MortalityDataset(train_samples, vocab_size, augment=True)
        val_dataset = MortalityDataset(val_samples, vocab_size, augment=False)
        
        # Create weighted sampler
        train_labels = [sample['label'] for sample in train_samples]
        class_counts = Counter(train_labels)
        weights = [1.0 / class_counts[label] for label in train_labels]
        sampler = WeightedRandomSampler(weights, len(weights))
        
        # Adjust batch size based on dataset size
        actual_batch_size = min(batch_size, len(train_samples) // 10)  # Ensure at least 10 batches
        actual_batch_size = max(2, actual_batch_size)  # Ensure batch size is at least 2
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=actual_batch_size, 
            sampler=sampler,
            drop_last=True  # Drop last incomplete batch
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=actual_batch_size, 
            drop_last=True
        )
        
        fold_best_model = None
        fold_best_val_score = 0
        
        # Try different hyperparameters
        for params in hyperparameters:
            model = MortalityPredictor(
                input_size=vocab_size,
                hidden_sizes=params['hidden_sizes'],
                dropout_rate=params['dropout_rate'],
                use_batch_norm=params['use_batch_norm']
            ).to(device)
            
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=params['lr'], 
                weight_decay=params['weight_decay']
            )
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=params['lr'],
                epochs=30,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,
                anneal_strategy='cos'
            )
            
            train_losses, val_losses, val_accuracies = train_model(
                model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=30
            )
            
            if val_accuracies[-1] > fold_best_val_score:
                fold_best_val_score = val_accuracies[-1]
                fold_best_model = copy.deepcopy(model)
        
        best_models.append(fold_best_model)
        best_val_scores.append(fold_best_val_score)
    
    # Return the best model across all folds
    best_fold = np.argmax(best_val_scores)
    return best_models[best_fold], best_val_scores[best_fold]

def main():
    logger.info("Starting mortality prediction pipeline...")

    try:
        # Initialize device first
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load dataset
        logger.info("Loading MIMIC3 dataset...")
        start_time = time.time()
        dataset = MIMIC3Dataset(
            root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD"],
            code_mapping={"ICD9CM": "CCSCM"},
            dev=True
        )
        logger.info(f"Dataset loaded successfully in {time.time() - start_time:.2f} seconds")
        
        # Create task dataset
        task_ds = dataset.set_task(task_fn=new_mortality_prediction_mimic3_fn)
        logger.info(f"Task dataset created with {len(task_ds.samples)} samples")
        
        # Analyze class distribution and calculate weights
        all_labels = [sample['label'] for sample in task_ds.samples]
        unique, counts = np.unique(all_labels), np.unique(all_labels, return_counts=True)[1]
        class_dist = dict(zip(unique, counts))
        
        logger.info("Class distribution in full dataset:")
        for label, count in class_dist.items():
            logger.info(f"Class {label}: {count} samples ({count/len(all_labels)*100:.2f}%)")
        
        if len(unique) <= 1:
            logger.error("Dataset contains only one class. Cannot train a meaningful classifier.")
            return
        
        # Calculate weights for imbalanced classes
        neg_count = class_dist[0]
        pos_count = class_dist[1]
        pos_weight = torch.tensor([neg_count / pos_count]).to(device)
        logger.info(f"Using positive class weight: {pos_weight.item():.2f}")
        
        # Split dataset
        train_samples, test_samples = train_test_split(task_ds.samples, test_size=0.2, random_state=42)
        train_samples, val_samples = train_test_split(train_samples, test_size=0.2, random_state=42)
        
        # Define vocabulary size and batch size
        vocab_size = 10000
        batch_size = min(8, len(train_samples) // 4)  # Ensure at least 4 batches
        logger.info(f"Using vocabulary size: {vocab_size}")
        logger.info(f"Using batch size: {batch_size}")
        
        # Create datasets with augmentation for training
        train_dataset = MortalityDataset(train_samples, vocab_size, augment=True)
        val_dataset = MortalityDataset(val_samples, vocab_size, augment=False)
        test_dataset = MortalityDataset(test_samples, vocab_size, augment=False)
        
        # Modified hyperparameters
        hyperparameters = [
            {
                'hidden_sizes': [64, 32],  # Smaller network
                'dropout_rate': 0.3,
                'use_batch_norm': True,
                'lr': 0.001,
                'weight_decay': 0.01
            },
            {
                'hidden_sizes': [128, 64],  # Medium network
                'dropout_rate': 0.4,
                'use_batch_norm': True,
                'lr': 0.0005,
                'weight_decay': 0.02
            },
            {
                'hidden_sizes': [256, 128],  # Larger network
                'dropout_rate': 0.5,
                'use_batch_norm': True,
                'lr': 0.0003,
                'weight_decay': 0.03
            }
        ]
        
        # Train with k-fold cross validation
        best_model, best_val_score = train_with_kfold(
            dataset=task_ds,
            vocab_size=vocab_size,
            device=device,
            hyperparameters=hyperparameters,
            batch_size=batch_size
        )
        logger.info(f"Best validation score across all folds: {best_val_score:.4f}")
        
        # Create final test loader
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Evaluate best model on test set
        test_metrics = evaluate_model(best_model, test_loader, device)
        
        logger.info("\nTest Set Metrics:")
        logger.info("Class distribution in test set:")
        for label, count in test_metrics['class_distribution'].items():
            logger.info(f"Class {label}: {count} samples")
        
        logger.info(f"\nAccuracy: {test_metrics['accuracy']:.4f}")
        
        if len(test_metrics['class_distribution']) > 1:
            logger.info(f"Precision: {test_metrics['precision']:.4f}")
            logger.info(f"Recall: {test_metrics['recall']:.4f}")
            logger.info(f"F1 Score: {test_metrics['f1']:.4f}")
            logger.info(f"AUC-ROC: {test_metrics['auc_roc']:.4f}")
        else:
            logger.warning("Some metrics are not reported due to single-class test set")
        
        # Plot confusion matrix
        plot_confusion_matrix(test_metrics['confusion_matrix'])

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    freeze_support()
    main()  
