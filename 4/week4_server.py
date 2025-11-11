# week4_server.py
# SERVER: Receives models and tracks communication metrics
# Run on Laptop/Server

import os
import sys
import time
import json
import socket
import pickle
import gzip
import threading
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
from fedma_utils import match_and_aggregate_layer

# ----------------------------
# Model Definition
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
# Deserialization Functions
# ----------------------------

def deserialize_baseline(model_bytes):
    """Deserialize baseline (full FP32)"""
    return pickle.loads(model_bytes)

def deserialize_fp16_gzip(model_bytes):
    """Deserialize FP16 + Gzip"""
    # Decompress
    decompressed = gzip.decompress(model_bytes)
    fp16_state = pickle.loads(decompressed)
    
    # Convert back to FP32
    fp32_state = {}
    for key, value in fp16_state.items():
        if value.dtype == torch.float16:
            fp32_state[key] = value.float()
        else:
            fp32_state[key] = value
    
    return fp32_state

def deserialize_fc_only(model_bytes, base_state):
    """Deserialize FC layers and merge with base model"""
    fc_state = pickle.loads(model_bytes)
    
    # Merge with base model
    full_state = base_state.copy() if base_state else {}
    full_state.update(fc_state)
    
    return full_state

def deserialize_delta(model_bytes, prev_state):
    """Deserialize delta updates"""
    delta_state = pickle.loads(model_bytes)
    
    if prev_state is None:
        return delta_state
    
    # Add delta to previous state
    full_state = {}
    for key in delta_state.keys():
        if key in prev_state:
            full_state[key] = prev_state[key] + delta_state[key]
        else:
            full_state[key] = delta_state[key]
    
    return full_state

# ----------------------------
# Server Class
# ----------------------------

class OptimizationServer:
    def __init__(self, host='0.0.0.0', port=9999):
        self.host = host
        self.port = port
        
        # Global model
        self.global_model = DisasterCNN(num_classes=4)
        self.global_state = self.global_model.state_dict()
        
        # Tracking
        self.communication_logs = []
        self.round_data = defaultdict(list)
        
        self.lock = threading.Lock()
        
        print("✅ Server initialized")
        print(f"   Global model parameters: {sum(p.numel() for p in self.global_model.parameters())}")
    
    def handle_client(self, conn, addr):
        """Handle incoming client connection"""
        print(f"\n📥 Connection from {addr}")
        
        try:
            # Receive metadata
            metadata_size = int.from_bytes(conn.recv(4), 'big')
            metadata_json = conn.recv(metadata_size).decode()
            metadata = json.loads(metadata_json)
            
            client_id = metadata['client_id']
            round_num = metadata['round']
            strategy = metadata['strategy']
            model_size = metadata['model_size']
            serialize_time = metadata['serialize_time']
            training_loss = metadata['training_loss']
            
            print(f"[Client {client_id}] Round {round_num}: {strategy} ({model_size/1024:.2f} KB)")
            
            # Receive model
            receive_start = time.time()
            
            model_bytes = b''
            remaining = model_size
            while remaining > 0:
                chunk = conn.recv(min(remaining, 8192))
                if not chunk:
                    break
                model_bytes += chunk
                remaining -= len(chunk)
            
            receive_time = time.time() - receive_start
            
            # Deserialize
            deserialize_start = time.time()
            
            if strategy == 'baseline':
                state_dict = deserialize_baseline(model_bytes)
            elif strategy == 'fp16_gzip':
                state_dict = deserialize_fp16_gzip(model_bytes)
            elif strategy == 'fc_only':
                state_dict = deserialize_fc_only(model_bytes, self.global_state)
            elif strategy == 'delta':
                state_dict = deserialize_delta(model_bytes, self.global_state)
            else:
                state_dict = deserialize_baseline(model_bytes)
            
            deserialize_time = time.time() - deserialize_start
            
            # Log metrics
            log_entry = {
                'round': round_num,
                'client_id': client_id,
                'strategy': strategy,
                'size_kb': model_size / 1024,
                'serialize_time_ms': serialize_time * 1000,
                'receive_time_s': receive_time,
                'deserialize_time_ms': deserialize_time * 1000,
                'training_loss': training_loss
            }
            
            with self.lock:
                # Store the log entry with state_dict for FedMA
                log_entry['round_complete'] = True
                log_entry['state_dict'] = state_dict  # Store the full state dict for FedMA
                self.communication_logs.append(log_entry)
                self.round_data[round_num].append(log_entry)
                
                # Update global model with FedMA
                client_models = []
                for log in self.communication_logs:
                    if log['round'] == round_num:
                        model = DisasterCNN(num_classes=4)
                        model.load_state_dict(log['state_dict'])
                        client_models.append(model)
                
                # Add current client's model
                model = DisasterCNN(num_classes=4)
                model.load_state_dict(state_dict)
                client_models.append(model)
                
                # Get all layer names that need to be matched
                layer_names = [
                    'features.0.weight', 'features.4.weight', 
                    'features.8.weight', 'features.12.weight',
                    'classifier.1.weight', 'classifier.4.weight'
                ]
                
                # Apply FedMA layer by layer
                new_global_state = self.global_state.copy()
                for layer_name in layer_names:
                    matched_weights, _ = match_and_aggregate_layer(
                        client_models, 
                        layer_name,
                        num_clients=len(client_models),
                        sigma_0=1.0,
                        sigma=1.0,
                        gamma_0=7.0
                    )
                    
                    if matched_weights is not None:
                        new_global_state[layer_name] = matched_weights
                        
                        # Also handle bias terms if they exist
                        bias_name = layer_name.replace('.weight', '.bias')
                        if bias_name in new_global_state:
                            # Simple average for biases
                            biases = [model.state_dict()[bias_name] for model in client_models 
                                    if bias_name in model.state_dict()]
                            if biases:
                                new_global_state[bias_name] = torch.mean(torch.stack(biases), dim=0)
                
                # Update global model
                self.global_state = new_global_state
                self.global_model.load_state_dict(self.global_state)
                
                # Log the update
                print(f"[Server] Global model updated with FedMA aggregation from {len(client_models)} clients")
            
            print(f"   ✅ Received & deserialized in {receive_time:.2f}s + {deserialize_time*1000:.2f}ms")
            print(f"   Loss: {training_loss:.4f}")
            
            # Send acknowledgment
            conn.sendall(b"OK")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            try:
                conn.sendall(f"Error: {e}".encode())
            except:
                pass
        
        finally:
            conn.close()
    
    def generate_reports(self):
        """Generate comparison reports and plots"""
        print(f"\n{'='*70}")
        print(f"📊 Generating Communication Optimization Reports")
        print(f"{'='*70}")
        
        if not self.communication_logs:
            print("No data collected")
            return
        
        os.makedirs('Week4_Server_Results', exist_ok=True)
        
        # Save CSV
        with open('Week4_Server_Results/communication_logs.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Round', 'Client', 'Strategy', 'Size_KB', 
                           'Serialize_ms', 'Receive_s', 'Deserialize_ms', 'Loss'])
            
            for log in self.communication_logs:
                writer.writerow([
                    log['round'],
                    log['client_id'],
                    log['strategy'],
                    f"{log['size_kb']:.2f}",
                    f"{log['serialize_time_ms']:.2f}",
                    f"{log['receive_time_s']:.2f}",
                    f"{log['deserialize_time_ms']:.2f}",
                    f"{log['training_loss']:.4f}"
                ])
        
        print("✅ Saved: Week4_Server_Results/communication_logs.csv")
        
        # Compute averages per strategy
        strategy_stats = defaultdict(lambda: {
            'sizes': [],
            'serialize_times': [],
            'deserialize_times': []
        })
        
        for log in self.communication_logs:
            strategy = log['strategy']
            strategy_stats[strategy]['sizes'].append(log['size_kb'])
            strategy_stats[strategy]['serialize_times'].append(log['serialize_time_ms'])
            strategy_stats[strategy]['deserialize_times'].append(log['deserialize_time_ms'])
        
        # Generate comparison table
        print(f"\n{'='*70}")
        print(f"📊 Strategy Comparison")
        print(f"{'='*70}")
        print(f"\n{'Strategy':<15} {'Avg Size':<12} {'Reduction':<12} {'Ser. Time':<12} {'Deser. Time':<12}")
        print("-"*70)
        
        baseline_size = np.mean(strategy_stats['baseline']['sizes']) if 'baseline' in strategy_stats else 0
        
        comparison_data = []
        
        for strategy in ['baseline', 'fp16_gzip', 'fc_only', 'delta']:
            if strategy in strategy_stats:
                stats = strategy_stats[strategy]
                avg_size = np.mean(stats['sizes'])
                avg_ser_time = np.mean(stats['serialize_times'])
                avg_deser_time = np.mean(stats['deserialize_times'])
                reduction = (1 - avg_size / baseline_size) * 100 if baseline_size > 0 else 0
                
                print(f"{strategy:<15} {avg_size:>10.2f} KB {reduction:>9.1f}% "
                      f"{avg_ser_time:>9.2f} ms {avg_deser_time:>9.2f} ms")
                
                comparison_data.append({
                    'strategy': strategy,
                    'size': avg_size,
                    'reduction': reduction,
                    'ser_time': avg_ser_time,
                    'deser_time': avg_deser_time
                })
        
        # Generate plots
        self.generate_plots(comparison_data)
        
        # Generate summary
        self.generate_summary(comparison_data)
    
    def generate_plots(self, comparison_data):
        """Generate comparison plots"""
        print(f"\n📊 Generating plots...")
        
        strategies = [d['strategy'] for d in comparison_data]
        sizes = [d['size'] for d in comparison_data]
        reductions = [d['reduction'] for d in comparison_data]
        ser_times = [d['ser_time'] for d in comparison_data]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        
        # Plot 1: Size comparison
        axes[0, 0].bar(range(len(strategies)), sizes, color=colors)
        axes[0, 0].set_xticks(range(len(strategies)))
        axes[0, 0].set_xticklabels(strategies, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Size (KB)')
        axes[0, 0].set_title('Model Size Comparison')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(sizes):
            axes[0, 0].text(i, v + 5, f'{v:.1f}', ha='center', va='bottom')
        
        # Plot 2: Reduction percentage
        axes[0, 1].bar(range(len(strategies)), reductions, color=colors)
        axes[0, 1].set_xticks(range(len(strategies)))
        axes[0, 1].set_xticklabels(strategies, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Reduction (%)')
        axes[0, 1].set_title('Size Reduction vs Baseline')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        for i, v in enumerate(reductions):
            axes[0, 1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        
        # Plot 3: Processing time
        axes[1, 0].bar(range(len(strategies)), ser_times, color=colors)
        axes[1, 0].set_xticks(range(len(strategies)))
        axes[1, 0].set_xticklabels(strategies, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Time (ms)')
        axes[1, 0].set_title('Serialization Time')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Size vs Time
        axes[1, 1].scatter(sizes, ser_times, s=200, c=colors, alpha=0.6)
        for i, strategy in enumerate(strategies):
            axes[1, 1].annotate(strategy, (sizes[i], ser_times[i]), 
                               fontsize=9, ha='right', va='bottom')
        axes[1, 1].set_xlabel('Size (KB)')
        axes[1, 1].set_ylabel('Serialize Time (ms)')
        axes[1, 1].set_title('Size vs Time Trade-off')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Week 4: Communication Optimization Results', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('Week4_Server_Results/optimization_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Saved: Week4_Server_Results/optimization_comparison.png")
    
    def generate_summary(self, comparison_data):
        """Generate text summary for report"""
        
        best_compression = min(comparison_data, key=lambda x: x['size'])
        best_speed = min(comparison_data, key=lambda x: x['ser_time'])
        
        summary = f"""
{'='*70}
WEEK 4: COMMUNICATION OPTIMIZATION SUMMARY
{'='*70}

1. BASELINE PERFORMANCE
   - Strategy: {comparison_data[0]['strategy']}
   - Size: {comparison_data[0]['size']:.2f} KB
   - Serialization Time: {comparison_data[0]['ser_time']:.2f} ms

2. BEST COMPRESSION
   - Strategy: {best_compression['strategy']}
   - Size: {best_compression['size']:.2f} KB
   - Reduction: {best_compression['reduction']:.1f}%
   - Time: {best_compression['ser_time']:.2f} ms

3. FASTEST PROCESSING
   - Strategy: {best_speed['strategy']}
   - Time: {best_speed['ser_time']:.2f} ms
   - Size: {best_speed['size']:.2f} KB

4. ALL STRATEGIES TESTED:
"""
        
        for data in comparison_data:
            summary += f"\n   {data['strategy']:<12}: {data['size']:>7.2f} KB ({data['reduction']:>5.1f}% reduction)"
        
        summary += f"""

5. RECOMMENDATION
   For UAV deployments:
   - Low Battery: Use fc_only (fastest, {comparison_data[2]['size']:.1f} KB)
   - Slow Network: Use fp16_gzip (smallest, {comparison_data[1]['size']:.1f} KB)
   - Balanced: Use delta (moderate size & speed)

{'='*70}
"""
        
        print(summary)
        
        with open('Week4_Server_Results/summary.txt', 'w') as f:
            f.write(summary)
        
        print("✅ Saved: Week4_Server_Results/summary.txt")
        
        # LaTeX table
        latex = r"""\begin{table}[htbp]
\centering
\caption{Communication Strategy Performance Comparison}
\label{tab:week4_optimization}
\begin{tabular}{lcccc}
\hline
\textbf{Strategy} & \textbf{Size (KB)} & \textbf{Reduction (\%)} & \textbf{Ser. Time (ms)} & \textbf{Deser. Time (ms)} \\
\hline
"""
        
        for data in comparison_data:
            latex += f"{data['strategy']} & "
            latex += f"{data['size']:.2f} & "
            latex += f"{data['reduction']:.1f} & "
            latex += f"{data['ser_time']:.2f} & "
            latex += f"{data['deser_time']:.2f} \\\\\n"
        
        latex += r"""\hline
\end{tabular}
\end{table}
"""
        
        with open('Week4_Server_Results/latex_table.tex', 'w') as f:
            f.write(latex)
        
        print("✅ Saved: Week4_Server_Results/latex_table.tex")
    
    def run(self):
        """Start server"""
        print("="*70)
        print("🖥️  Week 4 Server: Communication Optimization")
        print("="*70)
        print(f"Listening on {self.host}:{self.port}")
        print("Waiting for clients...")
        print("="*70)
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(10)
            
            try:
                while True:
                    conn, addr = s.accept()
                    thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                    thread.start()
            
            except KeyboardInterrupt:
                print("\n\n⏹️  Server stopped by user")
                self.generate_reports()

# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    server = OptimizationServer(host='0.0.0.0', port=9999)
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n\nGenerating final reports...")
        server.generate_reports()
        print("\n✅ Server shutdown complete")
