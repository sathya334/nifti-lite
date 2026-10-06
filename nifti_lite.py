import argparse
import csv
import gzip
import struct
from pathlib import Path

import numpy as np

HEADER_SIZE = 348

DATATYPES = {
    2: "uint8", 4: "int16", 8: "int32", 16: "float32",
    64: "float64", 256: "int8", 512: "uint16", 768: "uint32",
}
DATATYPE_CODES = {v: k for k, v in DATATYPES.items()}


def _open(path):
    path = str(path)
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def _parse_header(raw):
    if len(raw) < HEADER_SIZE:
        raise ValueError("file too small to be NIfTI")

    # The first 4 bytes must equal 348; whichever byte order makes that true wins.
    if struct.unpack("<i", raw[0:4])[0] == HEADER_SIZE:
        e = "<"
    elif struct.unpack(">i", raw[0:4])[0] == HEADER_SIZE:
        e = ">"
    else:
        raise ValueError("not a NIfTI-1 file (bad header size)")

    magic = raw[344:348].rstrip(b"\x00").decode("ascii", "replace")
    if magic not in ("n+1", "ni1"):
        raise ValueError(f"not a NIfTI-1 file (magic {magic!r})")

    dim = struct.unpack(e + "8h", raw[40:56])
    ndim = dim[0]
    datatype = struct.unpack(e + "h", raw[70:72])[0]
    pixdim = struct.unpack(e + "8f", raw[76:108])
    vox_offset, scl_slope, scl_inter = struct.unpack(e + "3f", raw[108:120])

    return {
        "shape": tuple(dim[1:ndim + 1]),
        "spacing": tuple(round(p, 4) for p in pixdim[1:min(ndim, 3) + 1]),
        "dtype": DATATYPES.get(datatype, f"unknown({datatype})"),
        "byte_order": "little" if e == "<" else "big",
        "vox_offset": int(vox_offset),
        "scl_slope": scl_slope,
        "scl_inter": scl_inter,
    }


def read_header(path):
    """Read only the 348-byte NIfTI-1 header (never loads the image data)."""
    with _open(path) as f:
        return _parse_header(f.read(HEADER_SIZE))


def load_data(path):
    """Load the full image as a numpy array, decoded from raw bytes."""
    with _open(path) as f:
        raw = f.read()
    hdr = _parse_header(raw[:HEADER_SIZE])
    if hdr["dtype"].startswith("unknown"):
        raise ValueError(f"unsupported data type {hdr['dtype']}")

    endian = "<" if hdr["byte_order"] == "little" else ">"
    dtype = np.dtype(hdr["dtype"]).newbyteorder(endian)
    count = int(np.prod(hdr["shape"]))
    data = np.frombuffer(raw, dtype=dtype, count=count, offset=hdr["vox_offset"])
    data = data.reshape(hdr["shape"], order="F")  # NIfTI stores x fastest

    if hdr["scl_slope"] not in (0.0, 1.0) or hdr["scl_inter"] != 0.0:
        data = data * hdr["scl_slope"] + hdr["scl_inter"]
    return data


def write_nifti(path, data, spacing=(1.0, 1.0, 1.0)):
    """Write a minimal NIfTI-1 file (used to make demo/test data)."""
    data = np.asarray(data)
    code = DATATYPE_CODES[str(data.dtype)]
    h = bytearray(HEADER_SIZE)
    struct.pack_into("<i", h, 0, HEADER_SIZE)
    dims = [data.ndim, *data.shape] + [1] * (7 - data.ndim)
    struct.pack_into("<8h", h, 40, *dims)
    struct.pack_into("<h", h, 70, code)
    struct.pack_into("<h", h, 72, data.dtype.itemsize * 8)
    pix = [1.0, *spacing] + [1.0] * (7 - len(spacing))
    struct.pack_into("<8f", h, 76, *pix)
    struct.pack_into("<3f", h, 108, 352.0, 1.0, 0.0)
    h[344:348] = b"n+1\x00"
    body = bytes(h) + b"\x00" * 4 + data.tobytes(order="F")
    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wb") as f:
        f.write(body)


def scan(folder):
    files = sorted(
        p for p in Path(folder).rglob("*")
        if p.name.endswith(".nii") or p.name.endswith(".nii.gz")
    )
    results = []
    for p in files:
        try:
            results.append((p, read_header(p), None))
        except Exception as err:
            results.append((p, None, str(err)))
    return results


def save_csv(results, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "shape", "spacing", "dtype", "error"])
        for path, hdr, err in results:
            if err:
                writer.writerow([path, "", "", "", err])
            else:
                writer.writerow([path, hdr["shape"], hdr["spacing"], hdr["dtype"], ""])


def cmd_scan(args):
    results = scan(args.folder)
    print(f"{'file':28} {'shape':18} {'spacing (mm)':20} dtype")
    print("-" * 78)
    shapes, spacings = set(), set()
    for path, hdr, err in results:
        if err:
            print(f"{path.name:28} ERROR: {err}")
            continue
        shapes.add(hdr["shape"])
        spacings.add(hdr["spacing"])
        print(f"{path.name:28} {str(hdr['shape']):18} {str(hdr['spacing']):20} {hdr['dtype']}")

    print(f"\n{len(results)} files scanned.")
    if len(shapes) > 1:
        print(f"WARNING: {len(shapes)} different shapes found: {sorted(shapes)}")
    if len(spacings) > 1:
        print(f"WARNING: {len(spacings)} different spacings found: {sorted(spacings)}")
    if len(shapes) <= 1 and len(spacings) <= 1:
        print("All files have consistent shape and spacing.")

    if args.csv:
        save_csv(results, args.csv)
        print(f"Saved results to {args.csv}")


def cmd_info(args):
    hdr = read_header(args.file)
    for key, value in hdr.items():
        print(f"{key:12} {value}")
    data = load_data(args.file)
    print(f"{'min':12} {data.min():.4g}")
    print(f"{'max':12} {data.max():.4g}")
    print(f"{'mean':12} {data.mean():.4g}")
    print(f"{'nonzero':12} {np.count_nonzero(data) / data.size:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Fast, dependency-light NIfTI reader.")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="scan a folder's headers (fast)")
    s.add_argument("folder")
    s.add_argument("--csv", help="save the results to a CSV file")
    s.set_defaults(func=cmd_scan)

    i = sub.add_parser("info", help="full details of one file, including image stats")
    i.add_argument("file")
    i.set_defaults(func=cmd_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


