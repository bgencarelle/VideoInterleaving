#!/usr/bin/env python3
"""Receive modem audio on a separate machine/process; see --help."""
import sys
from animation_modem.decoder import main

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, ValueError, OSError) as exc:
        sys.exit(str(exc))
