"""openpilot as the comma runs it, and the logs it writes.

`use_openpilot()` puts the openpilot submodule (vendor/openpilot) and its opendbc on the import
path and loads DrivingBench's patched drive_helpers (native/openpilot) in place of the stock one,
as the comma's deployment does: `device/native.py` calls its `clip_curvature` with
`max_curvature`, which stock openpilot lacks.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "vendor/openpilot"
DRIVE_HELPERS = "openpilot.selfdrive.controls.lib.drive_helpers"
PATCHED = REPO / "native/openpilot/selfdrive/controls/lib/drive_helpers.py"
SETUP = (
    "export GIT_LFS_SKIP_SMUDGE=1; git submodule update --init --depth 1 vendor/openpilot && "
    "git -C vendor/openpilot submodule update --init --depth 1 opendbc_repo"
)


def use_openpilot():
    """Import openpilot and opendbc from the submodule, with the patched drive_helpers."""
    if getattr(sys.modules.get(DRIVE_HELPERS), "__file__", None) == str(PATCHED):
        return
    packages = (VENDOR / "openpilot", VENDOR / "opendbc_repo/opendbc")
    if not all((p / "__init__.py").exists() for p in packages):
        raise ImportError(f"openpilot is not checked out; from the repository root run: {SETUP}")
    for path in (VENDOR / "opendbc_repo", VENDOR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    package, _, name = DRIVE_HELPERS.rpartition(".")
    parent = importlib.import_module(package)
    spec = importlib.util.spec_from_file_location(DRIVE_HELPERS, PATCHED)
    module = importlib.util.module_from_spec(spec)
    sys.modules[DRIVE_HELPERS] = module
    spec.loader.exec_module(module)
    setattr(parent, name, module)


def car_params(raw: bytes):
    """The logged CarParams, usable wherever openpilot expects CP."""
    import capnp

    car = VENDOR / "opendbc_repo/opendbc/car"
    capnp.remove_import_hook()
    with capnp.load(str(car / "car.capnp"), imports=[str(car)]).CarParams.from_bytes(raw) as cp:
        return cp.as_builder().as_reader()


def events(rlog: Path, kinds: set[str]):
    """(logMonoTime ns, kind, message) for each wanted message in one rlog, in log order."""
    import zstandard

    use_openpilot()
    from openpilot.cereal import log

    data = zstandard.ZstdDecompressor().stream_reader(rlog.open("rb")).read()
    for event in log.Event.read_multiple_bytes(data, traversal_limit_in_words=2**63 - 1):
        kind = event.which()
        if kind in kinds:
            yield event.logMonoTime, kind, getattr(event, kind)
