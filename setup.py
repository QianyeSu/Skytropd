from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py as _build_py

PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = PROJECT_ROOT / "skytropd"
RESOURCE_DIRS = ("ValidationData", "ValidationMetrics")
FORTRAN_SKIP_ENV = "SKYTROPD_SKIP_FORTRAN"


class build_py(_build_py):
    """Copy validation resources into the packaged wheel."""

    package_name = "skytropd"

    def run(self):
        super().run()
        build_package_dir = Path(self.build_lib) / self.package_name
        for resource_dir in RESOURCE_DIRS:
            src = PROJECT_ROOT / resource_dir
            dst = build_package_dir / resource_dir
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


class MesonBuildExt(build_ext):
    """Build the optional zero-crossing backend with Meson."""

    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        if os.environ.get(FORTRAN_SKIP_ENV, "").lower() in {"1", "true", "yes"}:
            self.announce(
                f"Skipping Fortran backend build because {FORTRAN_SKIP_ENV} is set",
                level=2,
            )
            return

        env = self._build_environment()
        self._require_tool("ninja", env)
        c_compiler = self._require_compiler(
            env,
            env_vars=("CC",),
            candidates=("cc", "clang", "gcc"),
            label="C compiler",
        )
        fortran_compiler = self._require_compiler(
            env,
            env_vars=("FC", "F77", "F90"),
            candidates=("gfortran",),
            label="Fortran compiler",
        )

        env["CC"] = c_compiler
        env["FC"] = fortran_compiler
        env["F77"] = fortran_compiler
        env["F90"] = fortran_compiler

        build_dir = Path(self.build_temp) / "meson-zero-crossing"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

        setup_cmd = [
            sys.executable,
            "-m",
            "mesonbuild.mesonmain",
            "setup",
            str(build_dir),
            str(PACKAGE_DIR),
            "--buildtype=release",
            "-Db_lto=true",
        ]
        self._run(setup_cmd, env)
        self._run(
            [sys.executable, "-m", "mesonbuild.mesonmain", "compile", "-C", str(build_dir)],
            env,
        )

        artifact = self._find_artifact(build_dir, "_zero_crossing_backend")
        target = Path(self.get_ext_fullpath(ext.name))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact, target)
        self.announce(f"Copied {artifact} -> {target}", level=2)

    def _build_environment(self):
        env = os.environ.copy()
        search_prefixes = []
        conda_prefix = env.get("CONDA_PREFIX")
        if conda_prefix:
            search_prefixes.append(Path(conda_prefix))
        search_prefixes.append(Path(sys.prefix))

        path_entries = env.get("PATH", "").split(os.pathsep) if env.get("PATH") else []
        for prefix in search_prefixes:
            for candidate in (prefix / "Scripts", prefix / "Library" / "bin", prefix / "bin"):
                if candidate.is_dir() and str(candidate) not in path_entries:
                    path_entries.insert(0, str(candidate))
        env["PATH"] = os.pathsep.join(path_entries)

        c_compiler = self._find_executable(env, ("CC",), ("cc", "clang", "gcc"))
        if c_compiler:
            env["CC"] = c_compiler

        fortran_compiler = self._find_executable(env, ("FC", "F77", "F90"), ("gfortran",))
        if fortran_compiler:
            env["FC"] = fortran_compiler
            env["F77"] = fortran_compiler
            env["F90"] = fortran_compiler

        return env

    def _find_executable(self, env, env_vars, candidates):
        for env_var in env_vars:
            configured = env.get(env_var)
            if not configured:
                continue
            resolved = shutil.which(configured, path=env["PATH"])
            if resolved:
                return resolved
            configured_path = Path(configured)
            if configured_path.is_file():
                return str(configured_path.resolve())

        for candidate in candidates:
            resolved = shutil.which(candidate, path=env["PATH"])
            if resolved:
                return resolved

            versioned = self._find_versioned_executable(candidate, env)
            if versioned:
                return versioned

        return None

    def _find_versioned_executable(self, prefix, env):
        for entry in env.get("PATH", "").split(os.pathsep):
            if not entry:
                continue
            path = Path(entry)
            if not path.is_dir():
                continue
            matches = sorted(
                candidate
                for candidate in path.glob(f"{prefix}-*")
                if candidate.is_file()
                and candidate.name.startswith(f"{prefix}-")
                and candidate.name[len(prefix) + 1 :].isdigit()
            )
            if matches:
                return str(matches[-1])

    def _require_tool(self, tool_name, env):
        if shutil.which(tool_name, path=env["PATH"]) is None:
            raise RuntimeError(
                f"required build tool '{tool_name}' was not found in PATH. "
                f"Install it or set {FORTRAN_SKIP_ENV}=1 to skip the backend."
            )

    def _require_compiler(self, env, env_vars, candidates, label):
        resolved = self._find_executable(env, env_vars, candidates)
        if resolved is None:
            configured = ", ".join(env_vars)
            searched = ", ".join(candidates)
            raise RuntimeError(
                f"required {label} was not found in PATH or via {configured}. "
                f"Searched for {searched}. Install it or set {FORTRAN_SKIP_ENV}=1 "
                "to skip the backend."
            )
        return resolved

    def _run(self, cmd, env):
        subprocess.run(cmd, check=True, env=env, cwd=PROJECT_ROOT)

    def _find_artifact(self, build_dir: Path, stem: str) -> Path:
        suffixes = (".pyd", ".so", ".dylib")
        matches = []
        for suffix in suffixes:
            matches.extend(build_dir.rglob(f"{stem}*{suffix}"))
        if not matches:
            raise RuntimeError(
                f"Meson build completed but no extension artifact matching {stem} was found"
            )
        return matches[0]


setup(
    cmdclass={"build_py": build_py, "build_ext": MesonBuildExt},
    ext_modules=[
        Extension(
            "skytropd._zero_crossing_backend",
            sources=[
                "skytropd/_zero_crossing_module.c",
                "skytropd/Fortran/zero_crossing.f90",
            ],
        )
    ],
)
