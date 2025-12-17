"""
Centralized server port configuration.

This module provides a single source of truth for all server port assignments,
eliminating magic numbers and making port changes easy even when nginx is configured.
"""
from dataclasses import dataclass
from typing import Optional
import settings


# Mode constants for type safety
MODE_WEB = "web"
MODE_LOCAL = "local"
MODE_ASCII = "ascii"
MODE_ASCIIWEB = "asciiweb"


@dataclass
class PortConfig:
    """Port configuration for a server mode."""
    monitor: int  # Web monitor dashboard port
    stream: Optional[int] = None  # MJPEG stream port (None if not used)
    ascii_telnet: Optional[int] = None  # ASCII telnet server port
    ascii_websocket: Optional[int] = None  # ASCII WebSocket server port
    ascii_monitor: Optional[int] = None  # ASCII stats monitor port

    def get_all_ports(self) -> list[int]:
        """Returns a list of all non-None ports in this configuration."""
        ports = []
        if self.monitor is not None:
            ports.append(self.monitor)
        if self.stream is not None:
            ports.append(self.stream)
        if self.ascii_telnet is not None:
            ports.append(self.ascii_telnet)
        if self.ascii_websocket is not None:
            ports.append(self.ascii_websocket)
        if self.ascii_monitor is not None:
            ports.append(self.ascii_monitor)
        return ports


class ServerConfig:
    """
    Centralized server port configuration manager.
    
    Provides mode-based port configurations with backward compatibility
    to existing settings.py values.
    """
    
    # Default port values (backward compatible with existing code)
    DEFAULT_MONITOR_PORT = 1978  # From BIRTH_YEAR
    DEFAULT_STREAM_PORT = 8080
    DEFAULT_ASCII_TELNET_PORT = 2323
    DEFAULT_ASCII_WEBSOCKET_PORT = 2424
    DEFAULT_LOCAL_PORT = 8888
    DEFAULT_ASCIIWEB_MONITOR_PORT = 1980

    def __init__(self):
        """Initialize with backward compatibility to settings.py."""
        self._current_mode: Optional[str] = None
        self._current_config: Optional[PortConfig] = None

    def set_mode(self, mode: str, primary_port: Optional[int] = None):
        """
        Set the current server mode and optional primary port override.
        
        Args:
            mode: One of MODE_WEB, MODE_LOCAL, MODE_ASCII, MODE_ASCIIWEB
            primary_port: Optional port override (used for ASCII modes)
        """
        self._current_mode = mode
        
        if mode == MODE_WEB:
            # Web mode: Monitor on 1978, Stream on 8080
            monitor = getattr(settings, 'WEB_PORT', self.DEFAULT_MONITOR_PORT)
            stream = getattr(settings, 'STREAM_PORT', self.DEFAULT_STREAM_PORT)
            self._current_config = PortConfig(
                monitor=monitor,
                stream=stream
            )
            
        elif mode == MODE_LOCAL:
            # Local mode: Monitor only on 8888
            monitor = getattr(settings, 'WEB_PORT', self.DEFAULT_LOCAL_PORT)
            self._current_config = PortConfig(
                monitor=monitor
            )
            
        elif mode == MODE_ASCII:
            # ASCII mode: Telnet on primary_port, Monitor on primary_port+1
            if primary_port is None:
                ascii_port = getattr(settings, 'ASCII_PORT', self.DEFAULT_ASCII_TELNET_PORT)
            else:
                ascii_port = primary_port
                
            # Monitor is always primary_port + 1 in ASCII mode (not from settings)
            monitor = ascii_port + 1
            self._current_config = PortConfig(
                monitor=monitor,
                ascii_telnet=ascii_port,
                ascii_monitor=monitor  # ASCII stats uses same port as monitor
            )
            
        elif mode == MODE_ASCIIWEB:
            # ASCIIWEB mode: Monitor on 1980, WebSocket on primary_port+1 (default 2424)
            monitor = getattr(settings, 'WEB_PORT', self.DEFAULT_ASCIIWEB_MONITOR_PORT)
            
            if primary_port is None:
                websocket_port = getattr(settings, 'WEBSOCKET_PORT', self.DEFAULT_ASCII_WEBSOCKET_PORT)
            else:
                websocket_port = primary_port + 1
                
            self._current_config = PortConfig(
                monitor=monitor,
                ascii_websocket=websocket_port
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_ports(self) -> PortConfig:
        """
        Get the current port configuration.
        
        Returns:
            PortConfig for the current mode
            
        Raises:
            RuntimeError: If set_mode() hasn't been called yet
        """
        if self._current_config is None:
            raise RuntimeError(
                "ServerConfig not initialized. Call set_mode() first, "
                "typically done in main.py configure_runtime()"
            )
        return self._current_config

    def get_monitor_port(self) -> int:
        """Get the monitor port for the current mode."""
        return self.get_ports().monitor

    def get_stream_port(self) -> Optional[int]:
        """Get the stream port for the current mode (None if not used)."""
        return self.get_ports().stream

    def get_ascii_telnet_port(self) -> Optional[int]:
        """Get the ASCII telnet port (None if not used in current mode)."""
        return self.get_ports().ascii_telnet

    def get_ascii_websocket_port(self) -> Optional[int]:
        """Get the ASCII WebSocket port (None if not used in current mode)."""
        return self.get_ports().ascii_websocket

    def get_ascii_monitor_port(self) -> Optional[int]:
        """Get the ASCII monitor port (None if not used in current mode)."""
        return self.get_ports().ascii_monitor


# Global instance - initialized by main.py
_config = ServerConfig()


def get_config() -> ServerConfig:
    """Get the global ServerConfig instance."""
    return _config


def get_ports() -> PortConfig:
    """Convenience function to get current port configuration."""
    return _config.get_ports()

