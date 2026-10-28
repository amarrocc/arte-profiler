"""Browser-mode macOS prototype, using the same resource collection as nicegui-pack."""

from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

root = Path(SPECPATH).parents[1]
sys.path.insert(0, str(root / "packaging" / "macos"))
from resources import package_resources

datas, binaries = package_resources(root)
datas += collect_data_files("nicegui")
datas += collect_data_files("colour")

a = Analysis(
    [str(root / "packaging" / "macos" / "launcher.py")],
    pathex=[str(root / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=collect_submodules("nicegui") + collect_submodules("imagecodecs")
    + ["imageio.plugins.tifffile_v3"],
    runtime_hooks=[str(root / "packaging" / "macos" / "runtime_hook.py")],
    excludes=["pytest", "tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6", "webview"],
    hooksconfig={"matplotlib": {"backends": ["Agg"]}},
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="Arte-Profiler",
    console=False, argv_emulation=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Arte-Profiler")
app = BUNDLE(
    coll, name="Arte-Profiler.app", bundle_identifier="org.arte-profiler.experimental",
    info_plist={
        "CFBundleShortVersionString": "0.1",
        "NSHighResolutionCapable": True,
    },
)
