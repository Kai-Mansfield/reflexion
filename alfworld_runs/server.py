import socket
import pickle
import torch

# Initialize the tensor on GPU
tensor = torch.tensor([0], device=0)
print("Tensor initialized on device:", tensor.device)

# Set up the server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("0.0.0.0", 12345))
server_socket.listen(1)
print("Server listening on port 12345")

def recv_all(conn, size):
    """Helper function to receive all data from the socket."""
    data = b''
    while len(data) < size:
        packet = conn.recv(size - len(data))
        if not packet:
            break
        data += packet
    return data

while True:  # Keep the server running
    # Accept a client connection
    conn, addr = server_socket.accept()
    print(f"Connection established: {addr}")

    try:
        while True:
            data = conn.recv(1024)  # Receive client data
            if not data:
                break  # No more data, break out of the loop

            # Decode the received data
            command = data.decode()
            print(f"Received command: {command}")

            if command == "GET tensor":
                # Send the current tensor to the client as a pickle-serialized object
                serialized_tensor = pickle.dumps(tensor)
                conn.sendall(len(serialized_tensor).to_bytes(4, 'big'))  # Send the length of the data first
                conn.sendall(serialized_tensor)  # Send the serialized tensor data
            elif command == "ADD 1":
                # Add 1 to the tensor and send the updated tensor back
                tensor += 1
                print(f"Tensor updated: {tensor}")
                serialized_tensor = pickle.dumps(tensor)
                conn.sendall(len(serialized_tensor).to_bytes(4, 'big'))  # Send the length of the data first
                conn.sendall(serialized_tensor)  # Send the serialized tensor data
            elif command == "SET tensor":
                # Receive the size of the incoming tensor
                tensor_size_data = conn.recv(4)
                tensor_size = int.from_bytes(tensor_size_data, 'big')

                # Receive the full tensor data
                tensor_data = recv_all(conn, tensor_size)

                # Deserialize the tensor data
                new_value = pickle.loads(tensor_data)  # Deserialize the tensor
                new_value = new_value.to(tensor.device)  # Update tensor with the new one
                print(f"Tensor replaced with: {new_value}")
                conn.sendall(b"Tensor updated successfully")
            elif command == "SET param":
                # Receive the size of the incoming parameter
                param_size_data = conn.recv(4)
                param_size = int.from_bytes(param_size_data, 'big')

                # Receive the full parameter data
                param_data = recv_all(conn, param_size)

                received_data = pickle.loads(param_data)
                param_cls = received_data['param_cls']
                old_value = received_data['old_value']
                device = received_data['device']

                new_value = param_cls(new_value, requires_grad=old_value.requires_grad).to(device)
                print(f"Parameter replaced with: {new_value}")
                conn.sendall(b"Parameter updated successfully")
            else:
                # Handle other commands if necessary
                conn.sendall(b"Unknown command")
    finally:
        # Clean up and close the connection after handling one client
        conn.close()
