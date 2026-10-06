from pathlib import Path

import numpy as np

from nifti_lite import write_nifti

out = Path("demo_scans")
out.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

# 5 normal scans: same shape and spacing
for i in range(5):
    write_nifti(out / f"patient_{i}.nii.gz",
                rng.random((64, 64, 40)).astype(np.float32), (1.0, 1.0, 1.0))

# 1 scan with a different shape, 1 with different spacing
write_nifti(out / "odd_shape.nii.gz",
            rng.random((64, 64, 32)).astype(np.float32), (1.0, 1.0, 1.0))
write_nifti(out / "odd_spacing.nii",
            rng.integers(0, 1000, (64, 64, 40)).astype(np.int16), (1.0, 1.0, 2.5))

# 1 broken file
(out / "broken.nii.gz").write_bytes(b"this is not an MRI")

print("Done: 8 demo files in demo_scans/")