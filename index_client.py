import socket
import json
import time
import threading

# Define the dictionary to store received data
midi_data_dictionary = {
    'Note_On': (0, 127, 0),
    'Note_Off': (None, None, None),
    'Modulation': (0, 0),
    'Index_and_Direction': (0, 1),
}


def receive_midi_data(server_address=('192.168.178.23', 12345)):
    global midi_data_dictionary  # specify that we're using the global dictionary

    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)  # Set a timeout of 5 seconds

    # Buffer for incoming data
    buffer = ''

    # Keep trying to connect to the server
    while True:
        try:
            sock.connect(server_address)
            print('Connected to server.')

            # Receive data
            while True:
                data = sock.recv(1024).decode()
                if data:
                    buffer += data
                    if '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        # Decode JSON data
                        json_data = json.loads(line)
                        # Store received data
                        midi_data_dictionary.update(json_data)
                        print(midi_data_dictionary['Index_and_Direction'])

        except (socket.error, socket.timeout) as e:  # Catch socket.timeout errors as well as socket.error errors
            print('Connection error:', e)
            print('Retrying in 5 seconds...')
            time.sleep(2)

        finally:
            sock.close()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)  # Set a timeout of 5 seconds for the new socket

# Start the MIDI data receiving loop in a separate thread

# threading.Thread(target=receive_midi_data, daemon=True).start()
