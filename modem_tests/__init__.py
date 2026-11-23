# The deprecated sender profiles (Fold 500, mono Fold 500, mono colour Fold
# 500, stereo slices) are no longer valid choices.  Their wires are still
# what the nested folds ride on and what the receiver must read, so their
# tests stay and this switch lets the tests select them.
import os as _os
_os.environ.setdefault('V7_ALLOW_DEPRECATED_PROFILES', '1')
