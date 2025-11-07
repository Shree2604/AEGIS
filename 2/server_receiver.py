# simple_server_receiver.py
# Run on SERVER (laptop) - Week 2 testing only

import socket
import json
import pickle
import threading

received_models = {}

def handle_client(conn, addr):
    """Handle incoming client connection"""
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
        model_state = pickle.loads(model_bytes)
        
        # Store received model
        key = (round_num, client_id)
        received_models[key] = {
            'state_dict': model_state,
            'metadata': metadata
        }
        
        print(f"[{client_id}] ✅ Model received successfully!")
        print(f"  - Training Loss: {metadata['training_loss']:.4f}")
        print(f"  - Training Time: {metadata['metrics']['duration_sec']:.2f}s")
        print(f"  - CPU Usage: {metadata['metrics']['avg_cpu_percent']:.1f}%")
        print(f"  - Temperature: {metadata['metrics']['end_temp_c']:.1f}°C")
        print(f"  - Battery Level: {metadata['battery_level']}%")
        
        # Send acknowledgment
        conn.sendall(b"Model received successfully!")
        
    except Exception as e:
        print(f"❌ Error handling client: {e}")
        conn.sendall(f"Error: {e}".encode())
    
    finally:
        conn.close()

def run_server(host='0.0.0.0', port=9999):
    """Run simple server to receive models"""
    print("="*60)
    print("🖥️  AEGIS Phase-3: Simple Server Receiver (Week 2 Testing)")
    print("="*60)
    print(f"Listening on {host}:{port}")
    print("Waiting for Pi clients...")
    print("="*60)
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(10)
        
        while True:
            conn, addr = s.accept()
            # Handle each client in a separate thread
            thread = threading.Thread(target=handle_client, args=(conn, addr))
            thread.start()

if __name__ == "__main__":
    run_server()
