from typing import Dict
import transformer_lens as lens
import torch
import torch.nn as nn
from general_utils import Metric


def linear_probe_across_layers(model: lens.HookedTransformer, 
                               features: Dict[int, torch.Tensor],
                               labels: torch.Tensor,
                               possible_label_count: int = 100,
                               train_test_split_percent: float = 0.8,
                               train_epochs: int = 20,
                               train_lr: float = 3e-4,
                               device: str = 'cuda',
                               verbose: bool = True,
                               ):
    """
    Perform a linear probing experiment across layers.
    The experiment trains a linear model on top of features (for every layer) to extract given labels,
    and measure the success rate.

    Args:
        model (lens.HookedTransformer): The TransformerLens model to probe.
        features (Dict[int, torch.Tensor]): The features extracted from the model to use as training and testing data for the probe.
        labels (torch.Tensor): The labels for the features.
        possible_label_count (int): The number of possible labels. Determines the output dimension of the linear probe.
        train_test_split_percent (float): The percentage of the data to use for training.
        train_epochs (int): The number of epochs to train the linear model.
        train_lr (float): The learning rate for the linear probe.
        device (str): The device to use for training the linear probe.
        verbose (bool): Whether to print accuracies and additional information during the training and testing of the linear probe.

    Returns:
        tuple(List(float), List(float)): The probing (train_accuracies, test_accuracies), where each list contains all layer accuracies.
    """
    probe_accs = ([], [])
    
    for layer_to_probe in features.keys():
        layer_features = features[layer_to_probe]

        # Define the probing datasets
        train_test_split = int(train_test_split_percent * len(layer_features))
        train_dataset = torch.utils.data.TensorDataset(layer_features[:train_test_split], labels[:train_test_split])
        test_dataset = torch.utils.data.TensorDataset(layer_features[train_test_split:], labels[train_test_split:])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Define the probing model, optimizer and loss
        linear_model = nn.Sequential(
            nn.Linear(model.cfg.d_model, possible_label_count),
        )
        linear_model.to(device)
        linear_model.requires_grad_(True)
        optimizer = torch.optim.Adam(linear_model.parameters(), lr=train_lr)
        loss_fn = nn.CrossEntropyLoss()
        
        # Training and testing loop
        with torch.set_grad_enabled(True):
            for epoch in range(train_epochs):
                acc = Metric()
                for batch_idx, (batch_features, batch_answers) in enumerate(train_loader):
                    optimizer.zero_grad()
                    batch_features, batch_answers = batch_features.to(device), batch_answers.to(device)
                    logits = linear_model(batch_features)
                    loss = loss_fn(logits, batch_answers)
                    acc.update((logits.argmax(dim=1) == batch_answers).float().mean().item())
                    loss.backward()
                    optimizer.step()
                
                with torch.no_grad():
                    test_acc = Metric()
                    for batch_idx, (batch_features, batch_answers) in enumerate(test_loader):
                        batch_features, batch_answers = batch_features.to(device), batch_answers.to(device)
                        logits = linear_model(batch_features)
                        test_acc.update((logits.argmax(dim=1) == batch_answers).float().mean().item())
                        
                if verbose:
                    print(f'Epoch {epoch+1}/{train_epochs}: Loss {loss.item():.3f}\t Train Accuracy: {acc.avg}\t Test Accuracy: {test_acc.avg :.3f}')

            if verbose:
                print(f'Layer {layer_to_probe}, Test Accuracy: {test_acc.avg :.3f}')

            probe_accs[0].append(acc.avg)
            probe_accs[1].append(test_acc.avg)

    return probe_accs
