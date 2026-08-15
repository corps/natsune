"""Custom build hooks for Hatchling to compile native extensions."""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface  # type: ignore


class CustomBuildHook(BuildHookInterface):
    """Custom build hook that compiles the atomic wrapper for macOS."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Initialize the build hook.

        This is called before each build and is where we compile
        the native extension for macOS so the files are available
        for inclusion in the wheel.
        """
        if sys.platform != "darwin":
            return

        root = Path(self.root)
        atomic_wrapper_dir = root / "src" / "atomic_wrapper"

        if not atomic_wrapper_dir.exists():
            return

        # Run make to compile the wrapper
        try:
            subprocess.run(
                ["make"],
                cwd=atomic_wrapper_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            self.app.display_warning(f"Failed to build atomic_wrapper: {e}")
            return

        # Ensure target directory exists
        target_dir = root / "src" / "natsune" / "lib" / "darwin"
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy the .dylib to the package
        dylib_src = atomic_wrapper_dir / "libatomic_wrapper.dylib"
        if dylib_src.exists():
            shutil.copy2(dylib_src, target_dir)
            self.app.display_info(f"Copied {dylib_src} to {target_dir}")

    def finalize(
        self, version: str, build_data: dict[str, Any], artifact_path: str
    ) -> None:
        """Finalize the build hook."""
        pass
