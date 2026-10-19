#!/usr/bin/env python3
"""Offline BNCFU W8/W4 weight formats, standard library only.

Input is a rectangular matrix of signed, unpacked INT8 values; INT4 mode
accepts only -8..7, still one signed byte per source weight, never nibbles.
There is no binary header. In both formats B=VLEN/8, planes are least
significant first, and bytes contain four 2-bit lane codes LSB-first.

Default qtype='2b' retains bncfu-radix4-v1 byte-for-byte: P=bits/2, T=4/P,
row-major [row][group][tile within group][plane][B/4 bytes], with
ceil(K/(B*T)) groups per row. For w, u=w+2**(bits-1),
digit[p]=((u>>(2*p))&3)-2 and code[p]=digit[p]&3. Missing lanes and tiles
encode original w=0 (NOT code=0). Thus w=sum(digit[p]*4**p)+42 for W8,
or +2 for W4. Each group occupies exactly B bytes.

qtype='1.5b' selects bncfu-ternary3-v1: P=6 for W8, P=3 for W4,
w=sum(digit[p]*3**p), digit in {-1,0,1}; no correction. Codes are
-1=>3, 0=>0, +1=>1; code 2 is illegal. Each row has ceil(K/B) tiles,
slots=tiles*P and ceil(slots/4) vectors laid out as
[row][vector][4 plane slots][B/4 bytes]. Slot s=tile*P+p has byte offset
row*row_bytes+(s//4)*B+(s%4)*(B/4)+lane//4, where
row_bytes=ceil(ceil(K/B)*P/4)*B. Missing K lanes and unused final slots
are all zero; no complete tile group is required. The ternary decoder
rejects illegal codes anywhere, nonzero padding, and reconstructed weights
outside the logical signed W8/W4 range (even if all digits are legal).
"""

import argparse
from collections.abc import Sequence
import json
from pathlib import Path


FORMAT_ID = "bncfu-radix4-v1"
LAYOUT = (
    "row-major [row][group][tile within group][plane][B/4 bytes]; "
    "B=vlen/8, P=bits/2, T=4/P; planes LSB-first; "
    "four 2-bit lane codes per byte LSB-first; padded original weights are zero"
)
TERNARY_FORMAT_ID = "bncfu-ternary3-v1"
TERNARY_LAYOUT = (
    "row-major [row][vector][4 plane slots][B/4 bytes]; B=vlen/8; "
    "P=6 for W8 or 3 for W4; slot=tile*P+plane; "
    "row_bytes=ceil(ceil(K/B)*P/4)*B; planes LSB-first, radix 3, no correction; "
    "four 2-bit lane codes per byte LSB-first (-1=3, 0=0, +1=1; 2 illegal); "
    "padded lanes and unused final slots are zero"
)


def _config(bits, vlen, qtype="2b"):
    if type(bits) is not int or bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    if (type(vlen) is not int or not 32 <= vlen <= 512
            or vlen & (vlen - 1)):
        raise ValueError("vlen must be a power of two from 32 through 512")
    if not isinstance(qtype, str) or qtype not in ("2b", "1.5b"):
        raise ValueError("qtype must be '2b' or '1.5b'")
    if qtype == "1.5b":
        return vlen // 8, 6 if bits == 8 else 3, None
    return vlen // 8, bits // 2, 8 // bits


def _row_bytes(rows, cols, bits, vlen, qtype="2b"):
    lanes, planes, tiles = _config(bits, vlen, qtype)
    for name, value in (("rows", rows), ("cols", cols)):
        if type(value) is not int or value < 0:
            raise ValueError(name + " must be a nonnegative integer")
    if qtype == "1.5b":
        slots = ((cols + lanes - 1) // lanes) * planes
        return ((slots + 3) // 4) * lanes
    return ((cols + lanes * tiles - 1) // (lanes * tiles)) * lanes


def _is_sequence(value):
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    )


def pack_weights(values, bits=8, vlen=256, qtype="2b"):
    """Return encoded bytes for a rectangular nested sequence of signed ints.

    bits is the logical weight width (4 or 8), not a packed digit width.
    qtype selects legacy radix-4 ('2b') or balanced ternary ('1.5b').
    Empty matrices and zero-length rows are supported. Invalid configuration,
    shape, noninteger weights, and out-of-range weights raise ValueError.
    """
    lanes, planes, tiles = _config(bits, vlen, qtype)
    if not _is_sequence(values):
        raise ValueError("values must be a nested sequence of signed integers")
    cols = None
    limit = 1 << (bits - 1)
    for row in values:
        if not _is_sequence(row):
            raise ValueError("each row must be a sequence of signed integers")
        if cols is None:
            cols = len(row)
        if len(row) != cols:
            raise ValueError("all rows must have the same number of columns")
        for weight in row:
            if type(weight) is not int or not -limit <= weight < limit:
                raise ValueError("weights must be integers in [%d, %d]"
                                 % (-limit, limit - 1))
    cols = 0 if cols is None else cols
    row_bytes = _row_bytes(len(values), cols, bits, vlen, qtype)
    if qtype == "1.5b":
        output = bytearray(len(values) * row_bytes)
        for row_index, row in enumerate(values):
            for col, weight in enumerate(row):
                tile, lane = divmod(col, lanes)
                for plane in range(planes):
                    digit = (weight + 1) % 3 - 1
                    weight = (weight - digit) // 3
                    slot = tile * planes + plane
                    offset = (row_index * row_bytes + (slot // 4) * lanes
                              + (slot % 4) * (lanes // 4) + lane // 4)
                    output[offset] |= (digit & 3) << (2 * (lane % 4))
        return bytes(output)
    output = bytearray()
    for row in values:
        for group in range(row_bytes // lanes):
            for tile in range(tiles):
                start = (group * tiles + tile) * lanes
                for plane in range(planes):
                    for lane in range(0, lanes, 4):
                        packed = 0
                        for field in range(4):
                            col = start + lane + field
                            weight = row[col] if col < cols else 0
                            digit = (((weight + limit) >> (2 * plane)) & 3) - 2
                            packed |= (digit & 3) << (2 * field)
                        output.append(packed)
    return bytes(output)


def unpack_weights(blob, rows, cols, bits, vlen, qtype="2b"):
    """Decode a raw blob, rejecting any length other than the exact row size.

    Only the requested rows and columns are returned; padding is discarded.
    blob must be bytes-like (bytes, bytearray, or memoryview). Ternary blobs
    additionally require legal codes, zero padding, and logical-range weights;
    legacy Q2 padding behavior is unchanged.
    """
    row_bytes = _row_bytes(rows, cols, bits, vlen, qtype)
    lanes, planes, tiles = _config(bits, vlen, qtype)
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise ValueError("blob must be bytes-like")
    blob = bytes(blob)
    if len(blob) != rows * row_bytes:
        raise ValueError("encoded length must be exactly %d bytes, got %d"
                         % (rows * row_bytes, len(blob)))
    if qtype == "1.5b":
        output = []
        slots = ((cols + lanes - 1) // lanes) * planes
        limit = 1 << (bits - 1)
        for row in range(rows):
            weights = [0] * cols
            for slot in range(row_bytes // (lanes // 4)):
                tile, plane = divmod(slot, planes)
                base = row * row_bytes + slot * (lanes // 4)
                for lane in range(lanes):
                    code = (blob[base + lane // 4] >> (2 * (lane % 4))) & 3
                    if code == 2:
                        raise ValueError("illegal ternary code 2")
                    col = tile * lanes + lane
                    if slot >= slots or col >= cols:
                        if code != 0:
                            raise ValueError("ternary padding must be zero")
                    else:
                        digit = -1 if code == 3 else code
                        weights[col] += digit * (3 ** plane)
            if any(not -limit <= weight < limit for weight in weights):
                raise ValueError("decoded ternary weight outside logical W%d range" % bits)
            output.append(weights)
        return output
    output = []
    for row in range(rows):
        weights = []
        for col in range(cols):
            group, within = divmod(col, lanes * tiles)
            tile, lane = divmod(within, lanes)
            base = row * row_bytes + group * lanes + tile * planes * (lanes // 4)
            unsigned = 0
            for plane in range(planes):
                byte = blob[base + plane * (lanes // 4) + lane // 4]
                code = (byte >> (2 * (lane % 4))) & 3
                digit = code if code < 2 else code - 4
                unsigned |= (digit + 2) << (2 * plane)
            weights.append(unsigned - (1 << (bits - 1)))
        output.append(weights)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path,
                        help="raw row-major signed unpacked bytes, NOT INT4 nibbles")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--bits", type=int, choices=(4, 8), required=True)
    parser.add_argument("--vlen", type=int, default=256)
    parser.add_argument("--qtype", choices=("2b", "1.5b"), default="2b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, help="optional JSON sidecar path")
    args = parser.parse_args(argv)
    try:
        row_bytes = _row_bytes(args.rows, args.cols, args.bits, args.vlen, args.qtype)
        _, planes, _ = _config(args.bits, args.vlen, args.qtype)
        raw = args.input.read_bytes()
        if len(raw) != args.rows * args.cols:
            raise ValueError("input length must be exactly rows*cols signed unpacked bytes")
        values = [
            [byte if byte < 128 else byte - 256
             for byte in raw[row * args.cols:(row + 1) * args.cols]]
            for row in range(args.rows)
        ]
        encoded = pack_weights(values, args.bits, args.vlen, args.qtype)
        args.output.write_bytes(encoded)
        if args.metadata is not None:
            ternary = args.qtype == "1.5b"
            metadata = dict(format=TERNARY_FORMAT_ID if ternary else FORMAT_ID,
                            bits=args.bits, vlen=args.vlen, qtype=args.qtype,
                            planes=planes, rows=args.rows, cols=args.cols,
                            row_bytes=row_bytes,
                            layout=TERNARY_LAYOUT if ternary else LAYOUT)
            args.metadata.write_text(json.dumps(metadata, indent=2) + "\n",
                                     encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
