"""Directory selection without initializing Tk inside a GLFW application."""
import json
from pathlib import Path
import subprocess
import sys


def pick_directory(current='', title='Select a folder'):
    """Return a selected path, or None on cancellation.

    macOS uses its native dialog through osascript. Other platforms run Tk in
    a separate interpreter so two GUI toolkits never share application state.
    """
    initial = str(Path(current).expanduser().resolve()) if current and Path(current).expanduser().is_dir() else ''
    if sys.platform == 'darwin':
        script = '''on run argv
    set dialogTitle to item 1 of argv
    if (count of argv) > 1 then
        set chosenFolder to choose folder with prompt dialogTitle default location (POSIX file (item 2 of argv))
    else
        set chosenFolder to choose folder with prompt dialogTitle
    end if
    return POSIX path of chosenFolder
end run'''
        command = ['/usr/bin/osascript', '-e', script, title]
        if initial:
            command.append(initial)
    else:
        command = [sys.executable, str(Path(__file__).resolve()), title, initial]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError(f'Could not open the folder picker: {exc}') from exc
    if result.returncode:
        if sys.platform == 'darwin' and '(-128)' in result.stderr:
            return None
        detail = result.stderr.strip() or f'dialog process exited with code {result.returncode}'
        raise RuntimeError(f'Could not open the folder picker: {detail}')
    if sys.platform == 'darwin':
        return result.stdout.rstrip('\r\n') or None
    try:
        selected = json.loads(result.stdout)
    except ValueError as exc:
        raise RuntimeError('Could not read the folder picker result.') from exc
    if selected is not None and not isinstance(selected, str):
        raise RuntimeError('Could not read the folder picker result.')
    return selected or None


def _tk_directory(title, initial):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    try:
        root.withdraw()
        try:
            root.attributes('-topmost', True)
        except tk.TclError:
            pass
        options = {'parent': root, 'title': title}
        if initial:
            options['initialdir'] = initial
        return filedialog.askdirectory(**options) or None
    finally:
        root.destroy()


if __name__ == '__main__':
    try:
        print(json.dumps(_tk_directory(sys.argv[1], sys.argv[2])))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
