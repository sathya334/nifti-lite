import struct

import numpy as np
import pytest

from nifti_lite import load_data, read_header, scan, write_nifti


@pytest.mark.parametrize("dtype", ["uint8", "int16", "int32", "float32", "float64"])
@pytest.mark.parametrize("ext", [".nii", ".nii.gz"])
def test_round_trip(tmp_path, dtype, ext):
    rng = np.random.default_rng(0)
    data = (rng.random((7, 5, 3)) * 100).astype(dtype)
    path = tmp_path / f"scan{ext}"
    write_nifti(path, data, spacing=(0.5, 0.5, 2.0))

    hdr = read_header(path)
    assert hdr["shape"] == (7, 5, 3)
    assert hdr["spacing"] == (0.5, 0.5, 2.0)
    assert hdr["dtype"] == dtype
    assert np.array_equal(load_data(path), data)


def test_axis_order_is_preserved(tmp_path):
    # Every voxel has a unique value, so any mix-up in x/y/z order would show.
    data = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    path = tmp_path / "order.nii"
    write_nifti(path, data)
    assert np.array_equal(load_data(path), data)


def test_big_endian_file(tmp_path):
    data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    h = bytearray(348)
    struct.pack_into(">i", h, 0, 348)
    struct.pack_into(">8h", h, 40, 3, 2, 3, 4, 1, 1, 1, 1)
    struct.pack_into(">h", h, 70, 4)       # int16
    struct.pack_into(">8f", h, 76, 1, 1, 1, 1, 1, 1, 1, 1)
    struct.pack_into(">3f", h, 108, 352.0, 1.0, 0.0)
    h[344:348] = b"n+1\x00"
    path = tmp_path / "big.nii"
    path.write_bytes(bytes(h) + b"\x00" * 4 + data.astype(">i2").tobytes(order="F"))

    assert read_header(path)["byte_order"] == "big"
    assert np.array_equal(load_data(path), data)


def test_garbage_file_raises(tmp_path):
    path = tmp_path / "broken.nii"
    path.write_bytes(b"not an mri" * 50)
    with pytest.raises(ValueError):
        read_header(path)


def test_scan_reports_errors_without_crashing(tmp_path):
    write_nifti(tmp_path / "good.nii.gz", np.zeros((4, 4, 4), dtype=np.float32))
    (tmp_path / "bad.nii").write_bytes(b"oops")
    results = {p.name: (hdr, err) for p, hdr, err in scan(tmp_path)}
    assert results["good.nii.gz"][1] is None
    assert results["bad.nii"][0] is None and results["bad.nii"][1]