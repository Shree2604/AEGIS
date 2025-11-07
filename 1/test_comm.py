# test_communication.py
# Run on BOTH Pi and Server to test connectivity

import socket
import json
import time

def test_server():
    """Run this on SERVER (laptop)"""
    HOST = '0.0.0.0'  # Listen on all interfaces
    PORT = 8888
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"✅ Server listening on {HOST}:{PORT}")
        print("Waiting for Pi client connection...")
        
        conn, addr = s.accept()
        with conn:
            print(f"✅ Connected by {addr}")
            
            # Receive message from client
            data = conn.recv(1024)
            message = json.loads(data.decode())
            print(f"📩 Received from client: {message}")
            
            # Send response
            response = {
                'status': 'success',
                'message': 'Server received your message!',
                'timestamp': time.time()
            }
            conn.sendall(json.dumps(response).encode())
            print("✅ Response sent to client")

def test_client(server_ip):
    """Run this on PI (client)"""
    HOST = server_ip  # Replace with your laptop's IP
    PORT = 8888
    
    print(f"Connecting to server at {HOST}:{PORT}...")
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("✅ Connected to server!")
        
        # Send message
        message = {
            'client_id': 'pi_test',
            'message': 'Hello from Raspberry Pi!',
            'timestamp': time.time()
        }
        s.sendall(json.dumps(message).encode())
        print(f"📤 Sent to server: {message}")
        
        # Receive response
        data = s.recv(1024)
        response = json.loads(data.decode())
        print(f"📩 Received from server: {response}")
        print("✅ Communication test successful!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  On Server: python test_communication.py server")
        print("  On Pi:     python test_communication.py client <server_ip>")
        print("\nExample: python test_communication.py client 192.168.1.100")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == 'server':
        test_server()
    elif mode == 'client':
        if len(sys.argv) < 3:
            print("❌ Error: Please provide server IP address")
            print("Example: python test_communication.py client 192.168.1.100")
            sys.exit(1)
        server_ip = sys.argv[2]
        test_client(server_ip)
    else:
        print("❌ Invalid mode. Use 'server' or 'client'")
        sys.exit(1)
