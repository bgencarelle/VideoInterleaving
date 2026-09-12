#!/bin/bash
# Finder-friendly macOS launcher; also usable from a Linux terminal.
cd "$(dirname "$0")" || exit 1
finish() {
    local result=$1
    if [ "$result" -ne 0 ]; then echo "Decoder setup failed (exit $result)." >&2; fi
    if [ -t 0 ] && [ "${MODEM_NO_PAUSE:-0}" != 1 ]; then
        read -r -p "Press Enter to close... " ignored
    fi
    exit "$result"
}
for argument in "$@"; do
    if [ "$argument" = --non-interactive ]; then MODEM_NO_PAUSE=1; fi
done
selected=""
if [ -n "${MODEM_PYTHON:-}" ]; then
    candidates=("$MODEM_PYTHON")
else
    candidates=(python3 /opt/local/bin/python3.12 /opt/local/bin/python3.11 python)
fi
for candidate in "${candidates[@]}"; do
    if "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))' >/dev/null 2>&1; then
        selected="$candidate"
        break
    fi
done
if [ -z "$selected" ]; then
    echo "Python 3.11+ was not found. Set MODEM_PYTHON to its executable path." >&2
    if [[ "$OSTYPE" == darwin* ]]; then
        echo "Older Macs: https://www.macports.org/install.php" >&2
        echo "After installing MacPorts: sudo port install python312" >&2
        echo "Then: MODEM_PYTHON=/opt/local/bin/python3.12 ./setup_decode.command" >&2
        if [ -t 0 ] && [ "${MODEM_NO_PAUSE:-0}" != 1 ]; then
            read -r -p "Open MacPorts installation instructions? [y/N] " answer
            case "$answer" in y|Y|yes|YES) open https://www.macports.org/install.php ;; esac
        fi
    else
        echo "Install Python 3 and its venv support using your OS package manager." >&2
    fi
    finish 1
fi
if [ "$#" -eq 0 ]; then set -- --install; fi
"$selected" setup_decode.py "$@"
finish "$?"
