import asyncio
import websockets
import json
import threading

# Define the dictionary to store received data
midi_data_dictionary = {
    'Note_On': (0, 127, 0),
    'Note_Off': (None, None, None),
    'Modulation': (0, 0),
    'Index_and_Direction': (0, 1),
}

async def receive_midi_data(uri="ws://192.168.178.23:12345"):
    global midi_data_dictionary  # specify that we're using the global dictionary
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print('Connected to server.')

                # Receive data
                while True:
                    data = await websocket.recv()
                    json_data = json.loads(data.decode())
                    # Store received data
                    midi_data_dictionary.update(json_data)
                    print(midi_data_dictionary['Index_and_Direction'])

        except (websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK) as e:
            print('Connection error:', e, flush=True)
            print('Retrying in 5 seconds...', flush=True)
            await asyncio.sleep(5)

def start_client():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    task = loop.create_task(receive_midi_data())
    loop.run_forever()

if __name__ == "__main__":
    # Start the client in a separate thread
    threading.Thread(target=start_client, daemon=True).start()
    # The rest of your script can continue here
