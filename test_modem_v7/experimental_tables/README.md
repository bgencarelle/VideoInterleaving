# Matched GUI test table

`chroma50.npz` is the opt-in nested GUI experiment's matched table, copied
verbatim from the live-validated chroma50 candidate. It is not a production
default or a generated animation bake.

- SHA-256: `2d4d7f097353cef1bd90755d48c17fbfbd3f26daf6e3ad4a5648011c704e0c3c`
- Luma: 120 rows × 100 columns; chroma: 60 rows × 50 columns.
- Both endpoints use the same table and explicitly enable the test patch.
- See `docs/V7_GUI_TEST_PATCH.md` for GUI settings and validation scope.

Reshaped (not reviewed by owner): the luma guests no longer run out in thin
strips along the first rows and columns (they drew full-width and
full-height lines: a cross on a Siemens star, streaks in a flat sky). Guests
now fill outward from the hosts by normalised radius, the second channel of
the two-channel tables taking the nearest positions as one region (its
first channel carries the riding details, nested_fold.DETAIL_GUESTS); guest
spreads from a smooth fit of the original table's, sectors by the builder's
rule; hosts, details, slots and bands unchanged. Made by tools/v7_reshape_guests.py from the original table.
