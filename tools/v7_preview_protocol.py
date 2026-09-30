"""Small standard-library-only protocol shared by V7 preview endpoints."""
import struct


MAGIC = b'V7EP'
HEADER = struct.Struct('!4sQIQ')


def pack_preview_datagram(counter, aspect, handoff_ns, jpeg):
    return (HEADER.pack(MAGIC, int(counter) & 0xffffffffffffffff,
                        int(aspect) & 0xffffffff, int(handoff_ns)) +
            bytes(jpeg))


def parse_preview_datagram(data):
    """Return packet counter, aspect code, handoff time, and JPEG bytes."""
    data = bytes(data)
    if len(data) <= HEADER.size:
        return None
    magic, counter, aspect, handoff_ns = HEADER.unpack_from(data)
    if magic != MAGIC:
        return None
    return counter, aspect, handoff_ns, data[HEADER.size:]
