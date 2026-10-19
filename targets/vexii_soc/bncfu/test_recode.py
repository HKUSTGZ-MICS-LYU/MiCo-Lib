#!/usr/bin/env python3
"""Host-only tests: python3 -B -m unittest discover -s <this directory>."""

import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from weight_recode import pack_weights, unpack_weights


VLENS = (32, 64, 128, 256, 512)
SCRIPT = Path(__file__).with_name("weight_recode.py")


def scalar_decode_tiles(blob, rows, cols, bits, vlen):
    """Independent stream consumer, including padded lanes, like a C kernel."""
    lanes = vlen // 8
    planes = bits // 2
    tiles = 4 // planes
    groups = (cols + lanes * tiles - 1) // (lanes * tiles)
    offset = 0
    decoded = []
    for _ in range(rows):
        row = []
        for _ in range(groups * tiles):
            digits = [0] * lanes
            for plane in range(planes):
                for first in range(0, lanes, 4):
                    byte = blob[offset]
                    offset += 1
                    for field in range(4):
                        code = (byte >> (field * 2)) & 3
                        digit = (0, 1, -2, -1)[code]
                        digits[first + field] += digit * (4 ** plane)
            row.extend(value + (42 if bits == 8 else 2) for value in digits)
        decoded.append(row)
    if offset != len(blob):
        raise AssertionError("scalar consumer did not consume the entire blob")
    return decoded


def ternary_digit_table(bits):
    """Independent oracle: enumerate digits, never use the packer's recurrence."""
    planes = 6 if bits == 8 else 3
    return {sum(d * 3 ** p for p, d in enumerate(digits)): digits
            for digits in itertools.product((-1, 0, 1), repeat=planes)}


def golden_ternary_pack(values, bits, vlen):
    """Assemble tile-plane chunks sequentially, with no slot offset formula."""
    lanes = vlen // 8
    table = ternary_digit_table(bits)
    output = bytearray()
    for row in values:
        chunks = bytearray()
        for start in range(0, len(row), lanes):
            padded = row[start:start + lanes] + [0] * max(0, start + lanes - len(row))
            for digits in zip(*(table[w] for w in padded)):
                for first in range(0, lanes, 4):
                    chunks.append(sum({-1: 3, 0: 0, 1: 1}[d] * 4 ** field
                                      for field, d in enumerate(digits[first:first + 4])))
        output.extend(chunks)
        output.extend(bytes((-len(chunks)) % lanes))
    return bytes(output)


def scalar_ternary_cursor(blob, rows, cols, bits, vlen):
    """Consume hardware-like vectors/cursors, retaining K padding and trace."""
    lanes = vlen // 8
    planes = 6 if bits == 8 else 3
    tiles = (cols + lanes - 1) // lanes
    vectors = (tiles * planes + 3) // 4
    decoded, traces = [], []
    for row in range(rows):
        vector_index = 0
        cursor = 0
        padded_row, trace = [], []
        for tile in range(tiles):
            weights = [0] * lanes
            for plane in range(planes):
                trace.append((tile, plane, vector_index, cursor))
                start = (row * vectors + vector_index) * lanes
                vector = blob[start:start + lanes]
                chunk = vector[cursor * (lanes // 4):(cursor + 1) * (lanes // 4)]
                if len(chunk) != lanes // 4:
                    raise AssertionError("incomplete vector slot")
                for byte_index, byte in enumerate(chunk):
                    for field in range(4):
                        code = (byte >> (2 * field)) & 3
                        if code == 2:
                            raise AssertionError("illegal code in scalar consumer")
                        weights[byte_index * 4 + field] += (0, 1, 0, -1)[code] * 3 ** plane
                cursor += 1
                if cursor == 4:
                    cursor = 0
                    vector_index += 1
            padded_row.extend(weights)
        if cursor:
            start = (row * vectors + vector_index) * lanes + cursor * (lanes // 4)
            end = (row + 1) * vectors * lanes
            if any(blob[start:end]):
                raise AssertionError("nonzero unused final slots")
        if any(padded_row[cols:]):
            raise AssertionError("nonzero K padding")
        decoded.append(padded_row)
        traces.append(trace)
    if len(blob) != rows * vectors * lanes:
        raise AssertionError("scalar consumer did not consume the entire blob")
    return decoded, traces


class RecodeTests(unittest.TestCase):
    def test_exhaustive_values(self):
        for bits in (4, 8):
            limit = 1 << (bits - 1)
            values = [list(range(-limit, limit))]
            for vlen in VLENS:
                with self.subTest(bits=bits, vlen=vlen):
                    blob = pack_weights(values, bits, vlen)
                    self.assertEqual(unpack_weights(blob, 1, 2 * limit, bits, vlen), values)
                    decoded = scalar_decode_tiles(blob, 1, 2 * limit, bits, vlen)
                    self.assertEqual(decoded[0][:2 * limit], values[0])
                    self.assertTrue(all(x == 0 for x in decoded[0][2 * limit:]))

    def test_int8_plane_layout_golden(self):
        # B=8: each plane has two bytes, each group exactly eight bytes.
        # u=0..7: low plane 4e 4e, next aa ff, upper planes aa aa.
        # u=128..135 differs only in the top plane, whose codes are zero.
        values = [list(range(-128, -120)) + list(range(8)), list(range(8)) * 2]
        negative = bytes.fromhex("4e 4e aa ff aa aa aa aa")
        positive = bytes.fromhex("4e 4e aa ff aa aa 00 00")
        expected = negative + positive + positive + positive
        self.assertEqual(pack_weights(values, 8, 64), expected)
        self.assertEqual(unpack_weights(expected, 2, 16, 8, 64), values)

    def test_int4_tile_and_padding_golden(self):
        # B=4, T=2; group 0 has two tiles, group 1 has one real zero
        # and seven padding zeros. Original zero encodes as aa 00 per tile.
        values = [[-8, -7, -6, -5, 4, 5, 6, 7, 0]]
        expected = bytes.fromhex("4e aa 4e 55 aa 00 aa 00")
        self.assertEqual(pack_weights(values, 4, 32), expected)
        self.assertEqual(unpack_weights(expected, 1, 9, 4, 32), values)

    def test_odd_shapes_padding_and_scalar_dots(self):
        rng = random.Random(0xB4CF0)
        for bits in (4, 8):
            limit = 1 << (bits - 1)
            for vlen in VLENS:
                lanes = vlen // 8
                tiles = 8 // bits
                for cols in (1, 3, lanes - 1, lanes + 1,
                             lanes * tiles - 1, lanes * tiles + 3):
                    with self.subTest(bits=bits, vlen=vlen, cols=cols):
                        values = [[rng.randrange(-limit, limit) for _ in range(cols)]
                                  for _ in range(3)]
                        blob = pack_weights(values, bits, vlen)
                        groups = (cols + lanes * tiles - 1) // (lanes * tiles)
                        self.assertEqual(len(blob), 3 * groups * lanes)
                        self.assertEqual(unpack_weights(blob, 3, cols, bits, vlen), values)
                        decoded = scalar_decode_tiles(blob, 3, cols, bits, vlen)
                        activations = [rng.randrange(-128, 128)
                                       for _ in range(groups * lanes * tiles)]
                        for original, padded in zip(values, decoded):
                            self.assertEqual(padded[:cols], original)
                            self.assertEqual(padded[cols:], [0] * (len(padded) - cols))
                            self.assertEqual(sum(a * w for a, w in zip(activations, padded)),
                                             sum(a * w for a, w in zip(activations, original)))

    def test_defaults_and_empty_shapes(self):
        self.assertEqual(pack_weights([[0]]), pack_weights([[0]], 8, 256))
        for bits in (4, 8):
            for vlen in VLENS:
                for values in ([], [[], [], []]):
                    self.assertEqual(pack_weights(values, bits, vlen), b"")
                    self.assertEqual(unpack_weights(b"", len(values), 0, bits, vlen), values)
                self.assertEqual(unpack_weights(b"", 0, 13, bits, vlen), [])
        self.assertEqual(pack_weights(((1, 2), (3, 4))), pack_weights([[1, 2], [3, 4]]))

    def test_invalid_shapes_and_values(self):
        for values in (None, 4, "abc", b"\0", [1], [[1], []], [[], [1]],
                       [[None]], [[1.0]], [[True]], [["1"]], [b"\0"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                pack_weights(values)
        for bits in (4, 8):
            limit = 1 << (bits - 1)
            for value in (-limit - 1, limit, 255):
                with self.subTest(bits=bits, value=value), self.assertRaises(ValueError):
                    pack_weights([[value]], bits)

    def test_invalid_configuration(self):
        for bits in (0, 2, 16, 4.0, "8", True, None):
            with self.subTest(bits=bits):
                with self.assertRaises(ValueError):
                    pack_weights([[0]], bits, 256)
                with self.assertRaises(ValueError):
                    unpack_weights(b"", 0, 0, bits, 256)
        for vlen in (0, -32, 16, 33, 96, 513, 1024, 32.0, "256", True, None):
            with self.subTest(vlen=vlen):
                with self.assertRaises(ValueError):
                    pack_weights([[0]], 8, vlen)
                with self.assertRaises(ValueError):
                    unpack_weights(b"", 0, 0, 8, vlen)

    def test_invalid_dimensions_and_encoded_size(self):
        for rows, cols in ((-1, 1), (1, -1), (1.0, 1), (1, 1.0),
                           (True, 1), (1, False), (None, 1), (1, "1")):
            with self.subTest(rows=rows, cols=cols), self.assertRaises(ValueError):
                unpack_weights(b"", rows, cols, 8, 256)
        for bits in (4, 8):
            blob = pack_weights([[0], [1], [2]], bits)
            for invalid in (blob[:-1], blob + b"\0", b""):
                with self.assertRaises(ValueError):
                    unpack_weights(invalid, 3, 1, bits, 256)
            self.assertEqual(unpack_weights(bytearray(blob), 3, 1, bits, 256), [[0], [1], [2]])
            self.assertEqual(unpack_weights(memoryview(blob), 3, 1, bits, 256), [[0], [1], [2]])
        for blob in (None, 32, "", []):
            with self.assertRaises(ValueError):
                unpack_weights(blob, 0, 0, 8, 256)
        with self.assertRaises(ValueError):
            unpack_weights(b"\0", 0, 0, 8, 256)


class TernaryRecodeTests(unittest.TestCase):
    def test_exhaustive_values_all_vlens(self):
        for bits in (4, 8):
            limit = 1 << (bits - 1)
            values = [list(range(-limit, limit))]
            for vlen in VLENS:
                with self.subTest(bits=bits, vlen=vlen):
                    blob = pack_weights(values, bits, vlen, "1.5b")
                    self.assertEqual(blob, golden_ternary_pack(values, bits, vlen))
                    self.assertEqual(unpack_weights(blob, 1, 2 * limit, bits, vlen, "1.5b"),
                                     values)
                    decoded, _ = scalar_ternary_cursor(blob, 1, 2 * limit, bits, vlen)
                    self.assertEqual(decoded[0][:2 * limit], values[0])
                    self.assertEqual(decoded[0][2 * limit:], [0] * (len(decoded[0]) - 2 * limit))

    def test_w8_plane_layout_golden(self):
        # B=4, P=6: tile 1 starts at cursor 2, tile 2 at cursor 0.
        # -128 = 1-3+9+27+81-243; 127 = 1-9-27-81+243.
        values = [[1, 3, 9, 27, -1, -3, -9, -27, 81, -81, 127, -128], [0] * 12]
        expected = bytes.fromhex(
            "01 04 10 40 00 00 03 0c 30 c0 00 00 50 c0 70 70 7d d0 00 00"
        ) + bytes(20)
        self.assertEqual(pack_weights(values, 8, 32, "1.5b"), expected)
        self.assertEqual(golden_ternary_pack(values, 8, 32), expected)
        self.assertEqual(unpack_weights(expected, 2, 12, 8, 32, "1.5b"), values)

    def test_w4_plane_layout_golden(self):
        # B=4, P=3: four tiles start at cursors 0,3,2,1 and need 3 vectors.
        # Final tile has one real lane (-8), with all remaining lanes zero.
        values = [[-8, -1, 0, 7, -3, 3, -7, 6, 1, -2, 2, -6, -8], [0] * 13]
        expected = bytes.fromhex("4d c0 43 30 d7 70 35 5c c0 01 00 03") + bytes(12)
        self.assertEqual(pack_weights(values, 4, 32, "1.5b"), expected)
        self.assertEqual(golden_ternary_pack(values, 4, 32), expected)
        self.assertEqual(unpack_weights(expected, 2, 13, 4, 32, "1.5b"), values)
        self.assertEqual(pack_weights([values[0][:4]], 4, 32, "1.5b"),
                         bytes.fromhex("4d c0 43 00"))

    def test_cursor_boundaries_and_minimal_row_sizes(self):
        for bits in (4, 8):
            planes = 6 if bits == 8 else 3
            table = ternary_digit_table(bits)
            limit = 1 << (bits - 1)
            for vlen in VLENS:
                lanes = vlen // 8
                for tiles in range(1, 10):
                    for tail in (1, lanes - 1, lanes):
                        cols = (tiles - 1) * lanes + tail
                        row = [((i * 7 + 3) % (2 * limit)) - limit for i in range(cols)]
                        values = [row, row[::-1]]
                        with self.subTest(bits=bits, vlen=vlen, tiles=tiles, tail=tail):
                            blob = pack_weights(values, bits, vlen, "1.5b")
                            vectors = (tiles * planes + 3) // 4
                            self.assertEqual(len(blob), 2 * vectors * lanes)
                            self.assertEqual(blob, golden_ternary_pack(values, bits, vlen))
                            decoded, traces = scalar_ternary_cursor(blob, 2, cols, bits, vlen)
                            for row_index, (original, padded, trace) in enumerate(
                                    zip(values, decoded, traces)):
                                self.assertEqual(padded[:cols], original)
                                for tile, plane, vector, cursor in trace:
                                    # Read from the traced hardware position and check
                                    # an independently enumerated digit, not a roundtrip.
                                    base = (row_index * vectors + vector) * lanes
                                    chunk = blob[base + cursor * (lanes // 4):
                                                 base + (cursor + 1) * (lanes // 4)]
                                    for lane in range(lanes):
                                        col = tile * lanes + lane
                                        weight = original[col] if col < cols else 0
                                        code = (chunk[lane // 4] >> (2 * (lane % 4))) & 3
                                        self.assertEqual(code, table[weight][plane] & 3)
                                self.assertEqual([entry[3] for entry in trace[::planes]],
                                                 [(tile * planes) % 4 for tile in range(tiles)])
        for bits, tiles, vectors in ((8, 1, 2), (8, 2, 3), (4, 1, 1), (4, 4, 3)):
            self.assertEqual(len(pack_weights([[0] * (tiles * 32)], bits, 256, "1.5b")),
                             vectors * 32)

    def test_odd_shapes_padding_and_random_dots(self):
        rng = random.Random(0x158B4CF0)
        for bits in (4, 8):
            limit = 1 << (bits - 1)
            planes = 6 if bits == 8 else 3
            for vlen in VLENS:
                lanes = vlen // 8
                for cols in (1, 3, lanes - 1, lanes, lanes + 1, 2 * lanes - 1,
                             2 * lanes, 2 * lanes + 1, 4 * lanes - 1,
                             4 * lanes, 4 * lanes + 1, 7 * lanes + 3):
                    with self.subTest(bits=bits, vlen=vlen, cols=cols):
                        values = [[rng.randrange(-limit, limit) for _ in range(cols)]
                                  for _ in range(3)]
                        blob = pack_weights(values, bits, vlen, "1.5b")
                        self.assertEqual(blob, golden_ternary_pack(values, bits, vlen))
                        self.assertEqual(unpack_weights(blob, 3, cols, bits, vlen, "1.5b"), values)
                        decoded, traces = scalar_ternary_cursor(blob, 3, cols, bits, vlen)
                        padded_cols = ((cols + lanes - 1) // lanes) * lanes
                        activations = [rng.randrange(-128, 128) for _ in range(padded_cols)]
                        for row_index, (original, padded, trace) in enumerate(zip(values, decoded, traces)):
                            self.assertEqual(padded[cols:], [0] * (padded_cols - cols))
                            reference = sum(a * w for a, w in zip(activations, original))
                            self.assertEqual(sum(a * w for a, w in zip(activations, padded)), reference)
                            row_bytes = len(blob) // 3
                            dot = 0
                            for tile, plane, vector, cursor in trace:
                                start = row_index * row_bytes + vector * lanes + cursor * (lanes // 4)
                                for lane in range(lanes):
                                    code = (blob[start + lane // 4] >> (2 * (lane % 4))) & 3
                                    dot += activations[tile * lanes + lane] * (0, 1, 0, -1)[code] * 3 ** plane
                            self.assertEqual(dot, reference)
                        slots = ((cols + lanes - 1) // lanes) * planes
                        row_bytes = ((slots + 3) // 4) * lanes
                        for row in range(3):
                            for slot in range(row_bytes // (lanes // 4)):
                                tile, _ = divmod(slot, planes)
                                for lane in range(lanes):
                                    if slot >= slots or tile * lanes + lane >= cols:
                                        offset = row * row_bytes + slot * (lanes // 4) + lane // 4
                                        self.assertEqual((blob[offset] >> (2 * (lane % 4))) & 3, 0)

    def test_zero_weights_and_empty_shapes(self):
        for bits in (4, 8):
            for vlen in VLENS:
                for values in ([], [[], [], []]):
                    self.assertEqual(pack_weights(values, bits, vlen, "1.5b"), b"")
                    self.assertEqual(unpack_weights(b"", len(values), 0, bits, vlen, "1.5b"), values)
                self.assertEqual(unpack_weights(b"", 0, 13, bits, vlen, "1.5b"), [])
                for cols in (1, vlen // 8 + 1, vlen // 2 + 1):
                    blob = pack_weights([[0] * cols] * 2, bits, vlen, "1.5b")
                    self.assertEqual(blob, bytes(len(blob)))
        self.assertEqual(pack_weights(((1, 2), (3, 4)), qtype="1.5b"),
                         pack_weights([[1, 2], [3, 4]], qtype="1.5b"))

    def test_invalid_qtypes(self):
        for qtype in ("1b", "ternary", "", "1.58b", None, True, 2, 1.5, [], {}, b"2b"):
            with self.subTest(qtype=qtype):
                with self.assertRaises(ValueError):
                    pack_weights([[0]], qtype=qtype)
                with self.assertRaises(ValueError):
                    unpack_weights(b"", 0, 0, 8, 256, qtype)

    def test_invalid_values_configs_dimensions_and_lengths(self):
        for values in (None, 4, "abc", b"\0", [1], [[1], []], [[], [1]],
                       [[None]], [[1.0]], [[True]], [["1"]], [b"\0"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                pack_weights(values, qtype="1.5b")
        for bits in (4, 8):
            limit = 1 << (bits - 1)
            for value in (-limit - 1, limit, 255):
                with self.assertRaises(ValueError):
                    pack_weights([[value]], bits, qtype="1.5b")
            for vlen in VLENS:
                blob = pack_weights([[0], [1], [-1]], bits, vlen, "1.5b")
                for invalid in (blob[:-1], blob + b"\0", b""):
                    with self.assertRaises(ValueError):
                        unpack_weights(invalid, 3, 1, bits, vlen, "1.5b")
                for view in (bytearray(blob), memoryview(blob)):
                    self.assertEqual(unpack_weights(view, 3, 1, bits, vlen, "1.5b"),
                                     [[0], [1], [-1]])
        for bits in (0, 2, 16, 4.0, "8", True, None):
            with self.assertRaises(ValueError):
                pack_weights([[0]], bits, 256, "1.5b")
            with self.assertRaises(ValueError):
                unpack_weights(b"", 0, 0, bits, 256, "1.5b")
        for vlen in (0, -32, 16, 33, 96, 513, 1024, 32.0, "256", True, None):
            with self.assertRaises(ValueError):
                pack_weights([[0]], 8, vlen, "1.5b")
            with self.assertRaises(ValueError):
                unpack_weights(b"", 0, 0, 8, vlen, "1.5b")
        for rows, cols in ((-1, 1), (1, -1), (1.0, 1), (1, 1.0),
                           (True, 1), (1, False), (None, 1), (1, "1")):
            with self.assertRaises(ValueError):
                unpack_weights(b"", rows, cols, 8, 256, "1.5b")
        for blob in (None, 32, "", [], b"\0"):
            with self.assertRaises(ValueError):
                unpack_weights(blob, 0, 0, 8, 256, "1.5b")

    def test_illegal_code_two_in_any_field_including_padding(self):
        for bits in (4, 8):
            for vlen in VLENS:
                original = pack_weights([[0]], bits, vlen, "1.5b")
                for byte in range(len(original)):
                    for field in range(4):
                        blob = bytearray(original)
                        blob[byte] = 2 << (2 * field)
                        with self.subTest(bits=bits, vlen=vlen, byte=byte, field=field):
                            with self.assertRaisesRegex(ValueError, "illegal ternary code 2"):
                                unpack_weights(blob, 1, 1, bits, vlen, "1.5b")

    def test_nonzero_padding_rejected(self):
        for bits in (4, 8):
            planes = 6 if bits == 8 else 3
            for vlen in VLENS:
                lanes = vlen // 8
                original = pack_weights([[0]], bits, vlen, "1.5b")
                for slot in range(len(original) // (lanes // 4)):
                    for lane in range(lanes):
                        if slot < planes and lane == 0:
                            continue
                        for code in (1, 3):
                            blob = bytearray(original)
                            blob[slot * (lanes // 4) + lane // 4] = code << (2 * (lane % 4))
                            with self.assertRaisesRegex(ValueError, "padding must be zero"):
                                unpack_weights(blob, 1, 1, bits, vlen, "1.5b")

    def test_all_legal_digit_combinations_enforce_logical_range(self):
        for bits in (4, 8):
            planes = 6 if bits == 8 else 3
            limit = 1 << (bits - 1)
            for weight, digits in ternary_digit_table(bits).items():
                # B=4 and lane=0 makes one byte per plane; other lanes zero.
                blob = bytes([d & 3 for d in digits] + [0] * ((-planes) % 4))
                with self.subTest(bits=bits, weight=weight):
                    if -limit <= weight < limit:
                        self.assertEqual(unpack_weights(blob, 1, 1, bits, 32, "1.5b"), [[weight]])
                        self.assertEqual(pack_weights([[weight]], bits, 32, "1.5b"), blob)
                    else:
                        with self.assertRaisesRegex(ValueError, "outside logical"):
                            unpack_weights(blob, 1, 1, bits, 32, "1.5b")

    def test_q2_defaults_remain_byte_identical(self):
        rng = random.Random(42)
        for bits in (4, 8):
            limit = 1 << (bits - 1)
            for vlen in VLENS:
                for cols in (1, 3, vlen // 8 + 1, vlen // 2 + 3):
                    values = [[rng.randrange(-limit, limit) for _ in range(cols)] for _ in range(2)]
                    default = pack_weights(values, bits, vlen)
                    self.assertEqual(default, pack_weights(values, bits, vlen, "2b"))
                    self.assertEqual(unpack_weights(default, 2, cols, bits, vlen),
                                     unpack_weights(default, 2, cols, bits, vlen, "2b"))
                    self.assertEqual(scalar_decode_tiles(default, 2, cols, bits, vlen)[0][:cols], values[0])


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable, "-B", str(SCRIPT), *map(str, args)],
                              capture_output=True, env=env, check=False)

    def test_binary_cli_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, metadata = (root / name for name in
                                        ("signed.bin", "weights.bin", "weights.json"))
            for bits in (4, 8):
                limit = 1 << (bits - 1)
                values = [[-limit, -1, 0, 1, limit - 1], [1, 0, -1, 0, 1], [0] * 5]
                source.write_bytes(bytes(w & 255 for row in values for w in row))
                for vlen in VLENS:
                    with self.subTest(bits=bits, vlen=vlen):
                        result = self.run_cli(source, "--rows", 3, "--cols", 5,
                                              "--bits", bits, "--vlen", vlen,
                                              "--output", output, "--metadata", metadata)
                        self.assertEqual(result.returncode, 0, result.stderr.decode())
                        self.assertEqual(result.stdout, b"")
                        self.assertEqual(output.read_bytes(), pack_weights(values, bits, vlen))
                        info = json.loads(metadata.read_text(encoding="utf-8"))
                        self.assertEqual(info["format"], "bncfu-radix4-v1")
                        for key, value in dict(rows=3, cols=5, bits=bits, vlen=vlen,
                                               qtype="2b", planes=bits // 2,
                                               row_bytes=len(output.read_bytes()) // 3).items():
                            self.assertEqual(info[key], value)
                        self.assertIn("[tile within group][plane]", info["layout"])
            result = self.run_cli(source, "--rows", 3, "--cols", 5, "--bits", 8,
                                  "--output", output)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual(output.read_bytes(), pack_weights(values, 8, 256))

    def test_ternary_cli_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, metadata = (root / name for name in
                                        ("signed.bin", "weights.bin", "weights.json"))
            for bits in (4, 8):
                limit = 1 << (bits - 1)
                for vlen in VLENS:
                    lanes = vlen // 8
                    for cols in (1, lanes, lanes + 1, 4 * lanes):
                        values = [[[-limit, -1, 0, 1, limit - 1][i % 5]
                                   for i in range(cols)], [0] * cols]
                        source.write_bytes(bytes(w & 255 for row in values for w in row))
                        with self.subTest(bits=bits, vlen=vlen, cols=cols):
                            result = self.run_cli(source, "--rows", 2, "--cols", cols,
                                                  "--bits", bits, "--vlen", vlen,
                                                  "--qtype", "1.5b", "--output", output,
                                                  "--metadata", metadata)
                            self.assertEqual(result.returncode, 0, result.stderr.decode())
                            self.assertEqual(result.stdout, b"")
                            self.assertEqual(output.read_bytes(), golden_ternary_pack(values, bits, vlen))
                            info = json.loads(metadata.read_text(encoding="utf-8"))
                            planes = 6 if bits == 8 else 3
                            row_bytes = ((((cols + lanes - 1) // lanes) * planes + 3) // 4) * lanes
                            for key, value in dict(format="bncfu-ternary3-v1", qtype="1.5b",
                                                   planes=planes, bits=bits, vlen=vlen,
                                                   rows=2, cols=cols, row_bytes=row_bytes).items():
                                self.assertEqual(info[key], value)
                            self.assertIn("[vector][4 plane slots]", info["layout"])
                            self.assertIn("radix 3, no correction", info["layout"])
                            self.assertIn("unused final slots are zero", info["layout"])
            for rows, cols in ((0, 0), (0, 13), (3, 0)):
                source.write_bytes(b"")
                result = self.run_cli(source, "--rows", rows, "--cols", cols,
                                      "--bits", 8, "--qtype", "1.5b", "--output", output,
                                      "--metadata", metadata)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                self.assertEqual(output.read_bytes(), b"")
                info = json.loads(metadata.read_text(encoding="utf-8"))
                self.assertEqual(info["row_bytes"], 64 if cols else 0)
                self.assertEqual(info["qtype"], "1.5b")
            # Explicit Q2 and omitted --qtype must agree in bytes and sidecar.
            source.write_bytes(b"\x80\x7f\x00")
            result = self.run_cli(source, "--rows", 1, "--cols", 3, "--bits", 8,
                                  "--qtype", "2b", "--output", output, "--metadata", metadata)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            explicit = output.read_bytes(), metadata.read_bytes()
            result = self.run_cli(source, "--rows", 1, "--cols", 3, "--bits", 8,
                                  "--output", output, "--metadata", metadata)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual((output.read_bytes(), metadata.read_bytes()), explicit)

    def test_cli_rejects_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for raw, extra in ((b"", []), (b"\0\0", []), (b"\x08", []),
                               (b"\xf7", []), (b"\0", ["--vlen", "96"]),
                               (b"\0", ["--rows", "-1"]),
                               (b"\0", ["--cols", "-1"]),
                               (b"\0", ["--bits", "2"]),
                               (b"\0", ["--qtype", "1b"]),
                               (b"\0", ["--qtype", "1.58b"]),
                               (b"\0", ["--qtype", "1.5b", "--vlen", "96"]),
                               (b"\x08", ["--qtype", "1.5b"]),
                               (b"\xf7", ["--qtype", "1.5b"])):
                source = root / "source.bin"
                output = root / "output.bin"
                source.write_bytes(raw)
                result = self.run_cli(source, "--rows", 1, "--cols", 1,
                                      "--bits", 4, "--output", output, *extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"error:", result.stderr)
                self.assertNotIn(b"Traceback", result.stderr)
                self.assertFalse(output.exists())
            # Nibble-packed INT4 is not accepted as two unpacked weights.
            source.write_bytes(b"\x17")
            result = self.run_cli(source, "--rows", 1, "--cols", 2,
                                  "--bits", 4, "--output", output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"signed unpacked bytes", result.stderr)
            result = self.run_cli(source)
            self.assertNotEqual(result.returncode, 0)
            result = self.run_cli(root / "missing.bin", "--rows", 1, "--cols", 1,
                                  "--bits", 8, "--output", output)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn(b"Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
