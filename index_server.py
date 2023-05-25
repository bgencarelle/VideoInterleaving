import socket
import json
import time
import midi_control  # assuming this is your module name
import threading

midi_control.midi_control_stuff_main()

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Get the current IP address
current_ip = socket.gethostbyname(socket.gethostname())
print('Current IP address:', current_ip)

# Bind the socket to a specific address and port
server_address = (current_ip, 12345)  # use the current IP address
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

# Keep track of the number of active connections
active_connections = 0
max_connections = 10
connection_lock = threading.Lock()


def handle_client(connection):
    global active_connections
    last_data = None
    try:
        # Send data
        while True:
            # Process MIDI
            midi_control.process_midi(2)
            data = midi_control.midi_data_dictionary
            encoded_data = (json.dumps(data) + '\n').encode()

            # Check if data has changed
            if encoded_data != last_data:
                # Send the MIDI data
                try:
                    connection.sendall(encoded_data)
                    last_data = encoded_data  # Update last_data
                except BrokenPipeError:
                    print("Client disconnected, stopping sending data")
                    break

            time.sleep(1 / 120)  # wait for 1/30th of a second
    finally:
        # Clean up the connection
        connection.close()
        with connection_lock:
            active_connections -= 1


while True:
    # Wait for a connection
    print('waiting for a connection')
    connection, client_address = sock.accept()

    with connection_lock:
        if active_connections >= max_connections:
            print('Too many connections, refusing connection from', client_address)
            connection.close()
        else:
            print('connection from', client_address)
            active_connections += 1
            thread = threading.Thread(target=handle_client, args=(connection,))
            thread.start()
