# web server thing for syncing
import asyncio
import websockets
import json
import time
import midi_control

MTC_CLOCK = 0
MIDI_CLOCK = 1
MIXED_CLOCK = 2
CLOCK_MODE = 2

midi_control.midi_control_stuff_main()

midi_data_dictionary = None


async def process_midi():
    global midi_data_dictionary
    while True:
        start_time = time.time()
        midi_control.process_midi(CLOCK_MODE)
        midi_data_dictionary = midi_control.midi_data_dictionary
        elapsed = time.time() - start_time
        sleep_time = max(1/90 - elapsed, 0)
        await asyncio.sleep(sleep_time)


async def handle_client(websocket, path):
    global midi_data_dictionary
    last_data = None
    try:
        while True:
            data = midi_data_dictionary
            encoded_data = (json.dumps(data) + '\n').encode()
            if encoded_data != last_data:
                try:
                    await websocket.send(encoded_data)
                    last_data = encoded_data
                except websockets.exceptions.ConnectionClosed:
                    print("Client disconnected, stopping sending data")
                    break
            await asyncio.sleep(1 / 90)
    except Exception as e:
        print(f"Exception: {e}")

start_server = websockets.serve(handle_client, "192.168.178.23", 12345)  # only works when ip is hardcoded for some

midi_process = asyncio.ensure_future(process_midi())
server = asyncio.ensure_future(start_server)

asyncio.get_event_loop().run_until_complete(server)
asyncio.get_event_loop().run_forever()
