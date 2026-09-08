"""PyInstaller entry point -- thin wrapper so the spec file has a plain
script to target (a `pip`-installed console-script entry point isn't
something PyInstaller can point at directly)."""

import sys

from book2audio.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
