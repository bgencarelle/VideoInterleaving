#!/usr/bin/env bash
# Run from the videointerleaving folder: shows why the kernel rows might be missing.
cd "$(dirname "$0")" 2>/dev/null; cd "${1:-.}"
echo "folder: $(pwd)"; ls dct_kernels/*.py 2>&1 | head
grep -c "dct_kernel" tools/v7_send_gui.py
.venv/bin/python - <<'P'
from tools import v7_send_gui as m
print("gui file:", m.__file__)
print("kernels:", m.kernel_registry(refresh=True).names())
g = m.SenderGui(())
print("direct DCT on:", g.settings['dct_encode'], "| pixel encode:", g.settings.get('pixel_encode'))
print("row visible:", 'dct_kernel' in g._visible_fields())
P