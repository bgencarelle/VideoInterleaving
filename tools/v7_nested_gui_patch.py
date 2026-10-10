"""Matched GUI launch settings for the opt-in nested codec experiment."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE_PATH = Path('test_modem_v7/experimental_tables/chroma50.npz')


def resolve_table_path(table_path=''):
    """Use an explicit path, or the GUI patch's bundled validated table."""
    selected = str(table_path or '').strip()
    table = Path(selected).expanduser() if selected else DEFAULT_TABLE_PATH
    if not table.is_absolute():
        table = ROOT/table
    if not table.is_file():
        raise ValueError(
            f'Matched nested table not found: {table}. Choose the bundled '
            'chroma50.npz at both endpoints.')
    return table.resolve()


def install_receiver(live, table_path, profile):
    """Install the same matched codec used by the receiver GUI Start action."""
    if profile not in ('aspect-mono-nested', 'stereo-nested'):
        raise ValueError('Matched codec test requires a nested wire profile.')
    table = resolve_table_path(table_path)
    from tools.v7_nested_experiment import install
    install(live, table)
    from tools.v7_adaptive_serpentine import install as install_codec
    install_codec(step_factor=0.25, exact_inverse=True)
    detail = profile == 'stereo-nested'
    if detail:
        os.environ['SLICE_CHROMA_DETAIL'] = '1'
    else:
        os.environ.pop('SLICE_CHROMA_DETAIL', None)
    return table, detail


def launcher_prefix(table_path, enabled, profile=None):
    """Return wrapper argv for an explicit experimental codec selection.

    An empty prefix preserves the ordinary CLI invocation byte-for-byte.
    """
    if not enabled:
        return []
    if profile not in ('aspect-mono-nested', 'stereo-nested'):
        raise ValueError('Matched codec test requires a nested wire profile.')
    table = resolve_table_path(table_path)
    root = ROOT
    result = [
        str(root/'tools'/'v7_nested_experiment.py'),
        '--tables', str(table),
        '--mapping', 'serpentine',
        '--adaptive-guest-range',
        '--step-factor', '0.25',
        '--exact-inverse',
    ]
    if profile == 'stereo-nested':
        result.append('--chroma-detail')
    return result


def wrap_sender(command, table_path, enabled, profile=None):
    """Wrap the regular sender command so the experiment settings precede send."""
    if not enabled:
        return command
    prefix = launcher_prefix(table_path, enabled, profile)
    # The experiment parser owns the command after its options.
    send_index = command.index('send')
    live_script = command[send_index-1]
    if not live_script.endswith('v7_live.py'):
        raise ValueError('Unexpected sender command; cannot install matched codec.')
    return [command[0], *prefix, command[send_index], *command[send_index+1:]]
