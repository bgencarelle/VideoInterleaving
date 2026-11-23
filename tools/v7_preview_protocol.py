"""Small standard-library-only protocol shared by V7 preview endpoints."""
import struct


MAGIC = b'V7IP'
HEADER = struct.Struct('!4sQIQB')
STAGE_CODES = {'source': 1, 'resized': 2}
CODE_STAGES = {code: stage for stage, code in STAGE_CODES.items()}


def pack_preview_datagram(counter, aspect, handoff_ns, stage, jpeg):
    try:
        stage_code = STAGE_CODES[str(stage)]
    except KeyError as exc:
        raise ValueError(f'unknown image preview stage: {stage}') from exc
    return (HEADER.pack(MAGIC, int(counter) & 0xffffffffffffffff,
                        int(aspect) & 0xffffffff, int(handoff_ns),
                        stage_code) +
            bytes(jpeg))


def parse_preview_datagram(data):
    """Return counter, aspect, handoff time, stage, and JPEG bytes."""
    data = bytes(data)
    if len(data) <= HEADER.size:
        return None
    magic, counter, aspect, handoff_ns, stage_code = HEADER.unpack_from(data)
    if magic != MAGIC:
        return None
    stage = CODE_STAGES.get(stage_code)
    if stage is None:
        return None
    return counter, aspect, handoff_ns, stage, data[HEADER.size:]
