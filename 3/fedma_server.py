# fedma_server.py
# Run on SERVER (laptop) - Complete FedMA aggregation with XAI

import os
import sys
import time
import json
import socket
import pickle
import threading
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from collections import defaultdict

# ----------------------------
# Import your existing model
# ----------------------------

class DisasterCNN(nn.Module):
    def __init__(self, num_classes, channel_sizes=None):
        super(DisasterCNN, self).__init__()
        
        if channel_sizes is None:
            channel_sizes = [16, 32, 64, 128]
        
        self.features = nn.Sequential(
            nn.Conv2d(3, channel_sizes[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(channel_sizes[0], channel_sizes[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(channel_sizes[1], channel_sizes[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(channel_sizes[2], channel_sizes[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_sizes[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(channel_sizes[3] * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self.channel_sizes = channel_sizes
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ----------------------------
# FedMA Functions (from your code)
# ----------------------------

def compute_channel_similarity(weight1, weight2, sigma_0=1.0, sigma=1.0):
    """Compute similarity between two CNN channels"""
    w1_flat = weight1.flatten()
    w2_flat = weight2.flatten()
    dist = torch.sum((w1_flat - w2_flat) ** 2).item()
    cost = dist / (2 * sigma ** 2) + 0.5 * torch.sum(w2_flat ** 2).item() / sigma_0 ** 2
    return cost

def hungarian_matching(local_weights_list, gamma_0=7.0, sigma_0=1.0, sigma=1.0):
    """Perform Hungarian matching for CNN channels"""
    num_clients = len(local_weights_list)
    global_channels = local_weights_list[0].clone()
    num_global_channels = global_channels.shape[0]
    
    permutation_matrices = []
    
    for client_id in range(num_clients):
        local_channels = local_weights_list[client_id]
        num_local_channels = local_channels.shape[0]
        
        max_size = num_global_channels + num_local_channels
        cost_matrix = np.ones((num_local_channels, max_size)) * 1e10
        
        for i in range(num_local_channels):
            for j in range(num_global_channels):
                cost = compute_channel_similarity(
                    local_channels[i], global_channels[j], sigma_0, sigma
                )
                cost_matrix[i, j] = cost
        
        epsilon = gamma_0
        for i in range(num_local_channels):
            for j in range(num_global_channels, max_size):
                cost_matrix[i, j] = epsilon + j * 0.01
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        new_global_channels = []
        assignment_map = {}
        
        for local_idx, global_idx in zip(row_ind, col_ind):
            assignment_map[local_idx] = global_idx
            
            if global_idx < num_global_channels:
                weight = 1.0 / (client_id + 1)
                new_channel = (1 - weight) * global_channels[global_idx] + weight * local_channels[local_idx]
                if global_idx < len(new_global_channels):
                    new_global_channels[global_idx] = new_channel
                else:
                    while len(new_global_channels) <= global_idx:
                        new_global_channels.append(global_channels[len(new_global_channels)] if len(new_global_channels) < num_global_channels else torch.zeros_like(local_channels[0]))
                    new_global_channels[global_idx] = new_channel
            else:
                new_global_channels.append(local_channels[local_idx].clone())
        
        if len(new_global_channels) > 0:
            for i in range(num_global_channels):
                if i >= len(new_global_channels):
                    new_global_channels.append(global_channels[i])
            
            global_channels = torch.stack(new_global_channels)
            num_global_channels = global_channels.shape[0]
        
        permutation_matrices.append(assignment_map)
    
    return global_channels, permutation_matrices

def match_and_aggregate_layer(local_models, layer_idx, sigma_0=1.0, sigma=1.0, gamma_0=7.0):
    """Match and aggregate a specific layer"""
    layer_name_patterns = [
        ('features.0.weight', 'conv'),
        ('features.4.weight', 'conv'),
        ('features.8.weight', 'conv'),
        ('features.12.weight', 'conv'),
        ('classifier.1.weight', 'fc'),
        ('classifier.4.weight', 'fc'),
    ]
    
    if layer_idx >= len(layer_name_patterns):
        return None, None
    
    layer_name, layer_type = layer_name_patterns[layer_idx]
    
    local_weights = []
    for model in local_models:
        state_dict = model.state_dict()
        if layer_name in state_dict:
            local_weights.append(state_dict[layer_name])
    
    if len(local_weights) == 0:
        return None, None
    
    if layer_type == 'conv':
        matched_weights, perm_matrices = hungarian_matching(
            local_weights, gamma_0=gamma_0, sigma_0=sigma_0, sigma=sigma
        )
    else:
        weights_stack = torch.stack(local_weights)
        matched_weights = torch.mean(weights_stack, dim=0)
        perm_matrices = None
    
    return matched_weights, perm_matrices

# ----------------------------
# FedMA Server
# ----------------------------

class FedMAServer:
    def __init__(self, num_classes=4, host='0.0.0.0', port=9999):
        self.num_classes = num_classes
        self.host = host
        self.port = port
        
        # Initialize global model
        self.global_model = DisasterCNN(num_classes=num_classes)
        
        # Round tracking
        self.current_round = 0
        self.received_models = defaultdict(dict)  # {round: {client_id: model_data}}
        self.round_metrics = []
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        print("✅ FedMA Server initialized")
        print(f"   Global model parameters: {sum(p.numel() for p in self.global_model.parameters())}")
    
    def handle_client(self, conn, addr):
        """Handle incoming client model"""
        print(f"\n📥 Connection from {addr}")
        
        try:
            # Receive metadata
            metadata_size = int.from_bytes(conn.recv(4), 'big')
            metadata_json = conn.recv(metadata_size).decode()
            metadata = json.loads(metadata_json)
            
            client_id = metadata['client_id']
            round_num = metadata['round']
            model_size = metadata['model_size']
            
            print(f"[{client_id}] Round {round_num}: Receiving {model_size/1024:.2f} KB...")
            
            # Receive model
            model_bytes = b''
            remaining = model_size
            while remaining > 0:
                chunk = conn.recv(min(remaining, 8192))
                if not chunk:
                    break
                model_bytes += chunk
                remaining -= len(chunk)
            
            # Deserialize model
            state_dict = pickle.loads(model_bytes)
            
            # Create model instance and load state
            local_model = DisasterCNN(num_classes=self.num_classes)
            local_model.load_state_dict(state_dict)
            
            # Store model
            with self.lock:
                self.received_models[round_num][client_id] = {
                    'model': local_model,
                    'metadata': metadata
                }
            
            print(f"[{client_id}] ✅ Model received - Loss: {metadata['training_loss']:.4f}")
            
            # Send acknowledgment
            conn.sendall(b"Model received successfully!")
            
        except Exception as e:
            print(f"❌ Error handling client: {e}")
            conn.sendall(f"Error: {e}".encode())
        
        finally:
            conn.close()
    
    def aggregate_round(self, round_num, expected_clients=4):
        """Perform FedMA aggregation for a round"""
        print(f"\n{'='*60}")
        print(f"🔄 Starting FedMA Aggregation - Round {round_num}")
        print(f"{'='*60}")
        
        # Wait for all clients (with timeout)
        max_wait = 60  # 60 seconds
        wait_start = time.time()
        
        while len(self.received_models[round_num]) < expected_clients:
            if time.time() - wait_start > max_wait:
                print(f"⚠️  Timeout: Only {len(self.received_models[round_num])}/{expected_clients} clients responded")
                break
            time.sleep(1)
        
        with self.lock:
            local_models = [data['model'] for data in self.received_models[round_num].values()]
            metadatas = [data['metadata'] for data in self.received_models[round_num].values()]
        
        if len(local_models) == 0:
            print("❌ No models received for aggregation")
            return
        
        print(f"✅ Aggregating {len(local_models)} client models...")
        
        # Perform FedMA layer-wise matching
        num_layers = 6
        new_global_state = {}
        
        for layer_idx in range(num_layers):
            print(f"   Matching layer {layer_idx + 1}/{num_layers}...")
            
            matched_weights, perm_matrices = match_and_aggregate_layer(
                local_models, layer_idx,
                sigma_0=1.0, sigma=1.0, gamma_0=7.0
            )
            
            if matched_weights is not None:
                layer_patterns = [
                    'features.0.weight',
                    'features.4.weight',
                    'features.8.weight',
                    'features.12.weight',
                    'classifier.1.weight',
                    'classifier.4.weight'
                ]
                layer_name = layer_patterns[layer_idx]
                new_global_state[layer_name] = matched_weights
                
                # Average biases
                bias_name = layer_name.replace('.weight', '.bias')
                bias_weights = []
                for model in local_models:
                    if bias_name in model.state_dict():
                        bias_weights.append(model.state_dict()[bias_name])
                
                if len(bias_weights) > 0:
                    new_global_state[bias_name] = torch.mean(torch.stack(bias_weights), dim=0)
        
        # Update global model
        global_state = self.global_model.state_dict()
        for key, value in new_global_state.items():
            if key in global_state:
                global_state[key] = value
        
        # Average other parameters (BatchNorm, etc.)
        for key in global_state.keys():
            if key not in new_global_state:
                param_list = []
                for model in local_models:
                    if key in model.state_dict():
                        param_list.append(model.state_dict()[key])
                
                if len(param_list) > 0:
                    if param_list[0].dtype == torch.long:
                        global_state[key] = param_list[0]
                    else:
                        global_state[key] = torch.mean(torch.stack(param_list), dim=0)
        
        self.global_model.load_state_dict(global_state, strict=False)
        
        # Compute metrics
        avg_loss = np.mean([m['training_loss'] for m in metadatas])
        total_upload_time = sum([m.get('metrics', {}).get('duration_sec', 0) for m in metadatas])
        
        print(f"\n✅ FedMA Aggregation Complete")
        print(f"   Avg Client Loss: {avg_loss:.4f}")
        print(f"   Total Training Time: {total_upload_time:.2f}s")
        
        # Save metrics
        self.round_metrics.append({
            'round': round_num,
            'num_clients': len(local_models),
            'avg_loss': avg_loss,
            'total_time': total_upload_time
        })
        
        # Save global model
        model_path = f"global_model_round_{round_num}.pth"
        torch.save(self.global_model.state_dict(), model_path)
        print(f"   Model saved: {model_path}")
    
    def run(self, num_rounds=5):
        """Main server loop"""
        print("="*60)
        print("🖥️  AEGIS Phase-3: FedMA Server")
        print("="*60)
        print(f"Listening on {self.host}:{self.port}")
        print(f"Total Rounds: {num_rounds}")
        print("="*60)
        
        # Start socket server in background thread
        def socket_server():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.host, self.port))
                s.listen(10)
                
                while self.current_round <= num_rounds:
                    conn, addr = s.accept()
                    thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                    thread.start()
        
        server_thread = threading.Thread(target=socket_server, daemon=True)
        server_thread.start()
        
        # Main round loop
        for round_num in range(1, num_rounds + 1):
            self.current_round = round_num
            
            print(f"\n{'='*60}")
            print(f"📡 Round {round_num}/{num_rounds} - Waiting for clients...")
            print(f"{'='*60}")
            
            # Wait for clients to send models (simulated wait)
            time.sleep(10)  # Adjust based on your training time
            
            # Perform aggregation
            self.aggregate_round(round_num)
            
            # TODO: Broadcast new global model back to clients
            # (Will implement in Week 4)
        
        print(f"\n{'='*60}")
        print(f"✅ Training Complete - {num_rounds} Rounds")
        print(f"{'='*60}")
        
        # Save final metrics
        self.save_metrics()
    
    def save_metrics(self):
        """Save training metrics"""
        os.makedirs("Phase3_Results", exist_ok=True)
        
        rounds = [m['round'] for m in self.round_metrics]
        losses = [m['avg_loss'] for m in self.round_metrics]
        
        plt.figure(figsize=(8, 6))
        plt.plot(rounds, losses, marker='o')
        plt.xlabel('Round')
        plt.ylabel('Average Loss')
        plt.title('FedMA Training Progress')
        plt.grid(True)
        plt.savefig('Phase3_Results/training_loss.png')
        plt.close()
        
        print("✅ Metrics saved to Phase3_Results/")

if __name__ == "__main__":
    server = FedMAServer(num_classes=4, host='0.0.0.0', port=9999)
    server.run(num_rounds=3)
