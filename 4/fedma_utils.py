import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_channel_similarity(weight1, weight2, sigma_0=1.0, sigma=1.0):
    """
    Compute similarity between two CNN channels using Gaussian posterior probability.
    Based on BBP-MAP from the FedMA paper.
    """
    # Flatten channel weights
    w1_flat = weight1.flatten()
    w2_flat = weight2.flatten()
    
    # Compute squared Euclidean distance
    dist = torch.sum((w1_flat - w2_flat) ** 2).item()
    
    # Gaussian posterior probability (negative log-likelihood as cost)
    cost = dist / (2 * sigma ** 2) + 0.5 * torch.sum(w2_flat ** 2).item() / sigma_0 ** 2
    
    return cost

def hungarian_matching(local_weights_list, gamma_0=7.0, sigma_0=1.0, sigma=1.0):
    """
    Perform Hungarian matching for CNN channels across multiple clients.
    
    Args:
        local_weights_list: List of weight tensors from different clients [C_out, C_in, H, W]
        gamma_0: Parameter controlling discovery of new channels
        sigma_0: Prior variance of global network weights
        sigma: Variance of local weights around global weights
    
    Returns:
        matched_global_weights: Aggregated global weights
        permutation_matrices: List of permutation info for each client
    """
    num_clients = len(local_weights_list)
    max_channels = max(w.shape[0] for w in local_weights_list)
    
    # Initialize global model with first client (will be updated iteratively)
    global_channels = local_weights_list[0].clone()
    num_global_channels = global_channels.shape[0]
    
    permutation_matrices = []
    
    for client_id in range(num_clients):
        local_channels = local_weights_list[client_id]
        num_local_channels = local_channels.shape[0]
        
        # Build cost matrix
        max_size = num_global_channels + num_local_channels
        cost_matrix = np.ones((num_local_channels, max_size)) * 1e10
        
        # Compute costs for matching with existing global channels
        for i in range(num_local_channels):
            for j in range(num_global_channels):
                cost = compute_channel_similarity(
                    local_channels[i], global_channels[j], sigma_0, sigma
                )
                cost_matrix[i, j] = cost
        
        # Add cost for creating new global channels
        for i in range(num_local_channels):
            for j in range(num_global_channels, max_size):
                cost_matrix[i, j] = gamma_0 + j * 0.01  # Small penalty for model size
        
        # Solve assignment problem using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Update global model based on matching
        new_global_channels = []
        assignment_map = {}
        
        for local_idx, global_idx in zip(row_ind, col_ind):
            assignment_map[local_idx] = global_idx
            
            if global_idx < num_global_channels:
                # Match to existing global channel - update with weighted average
                weight = 1.0 / (client_id + 1)
                new_channel = (1 - weight) * global_channels[global_idx] + weight * local_channels[local_idx]
                if global_idx < len(new_global_channels):
                    new_global_channels[global_idx] = new_channel
                else:
                    while len(new_global_channels) <= global_idx:
                        new_global_channels.append(global_channels[len(new_global_channels)] if len(new_global_channels) < num_global_channels else torch.zeros_like(local_channels[0]))
                    new_global_channels[global_idx] = new_channel
            else:
                # Create new global channel
                new_global_channels.append(local_channels[local_idx].clone())
        
        # Update global channels
        if len(new_global_channels) > 0:
            # Ensure we keep all existing channels that weren't matched
            for i in range(num_global_channels):
                if i >= len(new_global_channels):
                    new_global_channels.append(global_channels[i])
            
            global_channels = torch.stack(new_global_channels)
            num_global_channels = global_channels.shape[0]
        
        permutation_matrices.append(assignment_map)
    
    return global_channels, permutation_matrices

def match_and_aggregate_layer(local_models, layer_name, num_clients, sigma_0=1.0, sigma=1.0, gamma_0=7.0):
    """
    Match and aggregate a specific layer across all clients using FedMA.
    """
    # Extract weights from all clients for this layer
    local_weights = []
    for model in local_models:
        state_dict = model.state_dict()
        if layer_name in state_dict:
            local_weights.append(state_dict[layer_name])
    
    if len(local_weights) == 0:
        return None, None
    
    # Check if this is a conv or fc layer
    if 'features' in layer_name and 'weight' in layer_name:
        # For convolutional layers, match channels
        matched_weights, perm_matrices = hungarian_matching(
            local_weights, gamma_0=gamma_0, sigma_0=sigma_0, sigma=sigma
        )
    else:
        # For FC layers, use weighted averaging (simplified)
        weights_stack = torch.stack(local_weights)
        matched_weights = torch.mean(weights_stack, dim=0)
        perm_matrices = None
    
    return matched_weights, perm_matrices
