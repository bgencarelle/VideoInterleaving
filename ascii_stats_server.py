import socketserver
import time
import settings
from lightweight_monitor import monitor_data
from server_config import get_config

# --- CONFIGURATION ---
HOST = '127.0.0.1'


class ThreadedTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class StatsHandler(socketserver.BaseRequestHandler):
    def handle(self):
        # ANSI Codes (Strings)
        # \033[2J = Clear Screen
        # \033[H  = Move Cursor to Top-Left
        CLEAR_FRAME = "\033[2J\033[H"
        RESET = "\033[0m"
        BOLD = "\033[1m"
        GREEN = "\033[32m"
        CYAN = "\033[36m"
        RED = "\033[31m"

        try:
            while True:
                # 1. Fetch Data Snapshot
                d = monitor_data.copy()

                # 2. Determine Health Colors
                state = d.get('latency_state', 'unknown')
                if 'synced' in state:
                    state_color = GREEN
                elif 'ahead' in state or 'behind' in state:
                    state_color = RED
                else:
                    state_color = RESET

                # Format CPU
                cpu = d.get('cpu_percent', 0)
                cpu_str = f"{cpu:.1f}" if isinstance(cpu, (int, float)) else str(cpu)

                # 3. Build Dashboard
                output = [
                    f"{BOLD}=== ASCII STREAM MONITOR ==={RESET}",
                    f"Time: {time.strftime('%H:%M:%S')}",
                    f"",
                    f"{CYAN}[ PERFORMANCE ]{RESET}",
                    f" FPS:         {d.get('fps', 0)}",
                    f" Index:       {d.get('index', 0)}",
                    f" Delta:       {state_color}{d.get('delta', 0)} ({state}){RESET}",
                    f" FIFO Depth:  {d.get('fifo_depth', 0)}",
                    f" Staleness:   {d.get('staleness_ms', 0)} ms",
                    f"",
                    f"{CYAN}[ CONTENT ]{RESET}",
                    f" Main Folder: {d.get('main_folder', 0)}  ({d.get('main_covered', '0/0')})",
                    f" Float Folder:{d.get('float_folder', 0)}  ({d.get('float_covered', '0/0')})",
                    f" Entropy:     M:{d.get('main_entropy', 0)} / F:{d.get('float_entropy', 0)}",
                    f"",
                    f"{CYAN}[ SYSTEM ]{RESET}",
                    f" CPU:         {cpu_str}% ({d.get('threads', 0)} threads)",
                    f" Mem:         {d.get('mem_used', 0)} / {d.get('mem_total', 0)}",
                    f" Uptime:      {d.get('script_uptime', '0')}",
                    f"",
                    f"{CYAN}[ ERRORS ]{RESET}",
                    f" Failed Loads:{d.get('failed_load_count', 0)}",
                    f" Last Error:  {d.get('last_error', 'None')}",
                ]

                # 4. Construct Payload
                # Prepend CLEAR_FRAME to wipe the terminal clean every second
                text_block = "\n".join(output)
                full_payload = CLEAR_FRAME + text_block

                self.request.sendall(full_payload.encode('utf-8'))

                # 5. Refresh Rate (1Hz)
                time.sleep(1.0)

        except (BrokenPipeError, ConnectionResetError):
            pass  # Normal disconnect
        except Exception as e:
            print(f"[STATS] Error: {e}")


def start_server():
    port = get_config().get_ascii_monitor_port()
    if port is None:
        # Fallback to monitor port if ascii_monitor not set
        port = get_config().get_monitor_port()
    print(f"📊 Telnet Monitor started on port {port}")
    server = ThreadedTCPServer((HOST, port), StatsHandler)
    server.serve_forever()