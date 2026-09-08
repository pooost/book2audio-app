# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Book2Audio GUI. Onedir build (not onefile) --
onefile's runtime self-extraction is much more fragile with a dependency
stack this large (torch, transformers, spaCy, PySide6), and onedir starts
faster besides. Run from the repo root:

    pyinstaller packaging/book2audio.spec --noconfirm

Produces dist/Book2Audio/ containing the executable and all dependencies.
The CI workflow copies platform ffmpeg/ffprobe binaries into
dist/Book2Audio/ffmpeg-bin/ afterward -- see .github/workflows/build.yml.
"""

import sys

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

# These packages either ship non-Python data files (model configs, spaCy
# language data, Qt plugins) or use dynamic/plugin-style imports that
# PyInstaller's static analysis can't see on its own -- collect_all pulls in
# everything each package needs rather than discovering it via failed runs
# one at a time.
for pkg in (
    "torch",
    "torchaudio",
    "transformers",
    "chatterbox",
    "kokoro",
    "misaki",
    "spacy",
    "en_core_web_sm",
    "markitdown",
    "openocr",
    "PySide6",
    "huggingface_hub",
    "librosa",
    "soundfile",
):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["../run_gui.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Book2Audio",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="Book2Audio",
)

# macOS needs a real .app bundle (Finder-launchable, holds the Info.plist);
# Windows/Linux ship the onedir COLLECT() output directly.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Book2Audio.app",
        icon=None,
        bundle_identifier="com.book2audio.app",
        info_plist={"NSHighResolutionCapable": True},
    )
