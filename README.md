# nifti-lite

A small NIfTI reader I wrote from scratch, mainly to understand how MRI files are actually stored.

While working on brain tumor segmentation, I kept hitting the same problem: one scan in a dataset with a different shape or voxel spacing would quietly break preprocessing. Loading every full volume just to check its size is slow, so `nifti-lite` reads only the 348-byte header of each file and checks the whole dataset in seconds.

## Install

```bash
pip install nifti-lite
```

## Usage

Scan a folder of scans:

```bash
nifti-lite scan data/
nifti-lite scan data/ --csv results.csv
```

```
file                         shape              spacing (mm)         dtype
------------------------------------------------------------------------------
broken.nii.gz                ERROR: Not a gzipped file
odd_shape.nii.gz             (64, 64, 32)       (1.0, 1.0, 1.0)      float32
patient_0.nii.gz             (64, 64, 40)       (1.0, 1.0, 1.0)      float32
...
WARNING: 2 different shapes found: [(64, 64, 32), (64, 64, 40)]
```

Look at one file in detail, including image statistics:

```bash
nifti-lite info scan.nii.gz
```

Or use it from Python:

```python
from nifti_lite import read_header, load_data

hdr = read_header("scan.nii.gz")   # header only, fast
data = load_data("scan.nii.gz")    # full volume as a numpy array
```

## How it works

A NIfTI-1 file starts with a fixed 348-byte header. `nifti-lite` unpacks it with Python's `struct` module: dimensions at byte 40, data type at byte 70, voxel spacing at byte 76, and so on.

The byte order isn't stored as a flag, so it's detected by trick: the first 4 bytes always hold the number 348, and whichever byte order reads them as 348 is the right one.

To load the image, it reads the raw bytes after the header, decodes them with the correct data type and byte order, and reshapes them in column-major order, since NIfTI stores the x axis fastest. Intensity scaling (slope/intercept) is applied if the file uses it.

For `.nii.gz` files, scanning only decompresses the first few hundred bytes, which is why it's fast even on big datasets.

## Tests

```bash
pip install pytest
pytest -v
```

The tests write files and read them back for 5 data types in both `.nii` and `.nii.gz`, check that axis order is preserved, and include a hand-built big-endian file.

## Limitations

Supports NIfTI-1 only (not NIfTI-2), and doesn't apply the orientation matrix. For full-featured work, use [nibabel](https://nipy.org/nibabel/). This project is about understanding the format and fast dataset checks.

## License

MITp