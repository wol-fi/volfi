"""Mechanically transform volfi_annulus_tables.hpp into the CUDA device table header.

Companion to make_device_constants.py.  The device table header was originally produced
by hand; this script reproduces it exactly so it can be REGENERATED whenever the host
table changes (v0.2.3 added 57 deep cells, which is what motivated writing it).

Transform, and nothing else:
  * every top-level `static const` becomes `__device__ static const`
  * namespace volfi_annulus            -> volfi_annulus_gpu
  * include guard VOLFI_ANNULUS_TABLES_HPP -> VOLFI_ANNULUS_TABLES_CUDA_CUH
Numeric text is copied verbatim, so the device tables are bit-identical to the host
ones by construction.

Usage:  python make_device_tables.py      (from the gpu/ folder)
"""
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# ../include/volfi in the published repo, .. in the development tree.
SRC = next((c for c in (_HERE.parent / "include" / "volfi" / "volfi_annulus_tables.hpp",
                        _HERE.parent / "volfi_annulus_tables.hpp") if c.is_file()), None)
if SRC is None:
    raise SystemExit("cannot locate volfi_annulus_tables.hpp "
                     "(looked in ../include/volfi and ..)")
DST = Path(__file__).resolve().parent / "volfi_annulus_tables_cuda.cuh"

src = SRC.read_text(encoding="utf-8", errors="replace")

n_before = len(re.findall(r"^static const", src, re.M))
out = re.sub(r"^static const", "__device__ static const", src, flags=re.M)
out = out.replace("namespace volfi_annulus {", "namespace volfi_annulus_gpu {")
out = out.replace("} // namespace volfi_annulus", "} // namespace volfi_annulus_gpu")
out = out.replace("VOLFI_ANNULUS_TABLES_HPP", "VOLFI_ANNULUS_TABLES_CUDA_CUH")
n_after = len(re.findall(r"^__device__ static const", out, re.M))

if n_before != n_after or n_before == 0:
    sys.exit("transform mismatch: %d static const -> %d __device__" % (n_before, n_after))
if "namespace volfi_annulus {" in out:
    sys.exit("namespace rename failed")

DST.write_text(out, encoding="utf-8")
print("wrote %s  (%d declarations, %d bytes)" % (DST.name, n_after, len(out)))
