import socket
import pickle
import torch

# Set up the client to connect to the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("artemis-a40-12", 12345))  # Replace with the server's address and port

# Send a command to the server (GET tensor, ADD 1, or SET tensor)
command = "SET tensor"  # Command to set a new tensor
s.sendall(command.encode())  # Send command to the server

# Create new tensor to send (example size and values)
new_tensor = torch.tensor([10, 20, 30], device=0)

# Serialize the tensor
tensor_data = pickle.dumps(new_tensor)

# Send the length of the data first
s.sendall(len(tensor_data).to_bytes(4, 'big'))

# Send the serialized tensor data
s.sendall(tensor_data)

# Receive a response from the server
response = s.recv(1024)
print(response.decode())  # Print the server's response

# Clean up the connection
s.close()
