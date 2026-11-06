"""Создание окружения без системного ensurepip; только стандартная библиотека."""
from collections import deque
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen

from .io import atomic_write, file_sha

VIRTUALENV_VERSION = "21.15.5"
VIRTUALENV_URL = f"https://github.com/pypa/virtualenv/releases/download/{VIRTUALENV_VERSION}/virtualenv.pyz"
VIRTUALENV_SHA256 = "5363ec270906e8fce9fb6700012b6a243454bebe932013e4ee24ead97e504d11"
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / ".cache/bootstrap"


def run_command(command, cwd=None):
    """Передаёт stdout/stderr ячейке, сохраняя диагностический хвост ошибки."""
    command = [str(part) for part in command]
    tail = deque(maxlen=20)
    with subprocess.Popen(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            tail.append(line)
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f"Command failed ({returncode}): {' '.join(command)}\n{''.join(tail)}")


def environment_python(directory):
    return Path(directory) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def get_virtualenv(cache_dir=DEFAULT_CACHE):
    archive = Path(cache_dir) / f"virtualenv-{VIRTUALENV_VERSION}.pyz"
    if archive.exists() and file_sha(archive) == VIRTUALENV_SHA256:
        return archive
    with urlopen(VIRTUALENV_URL, timeout=60) as response:
        body = response.read(20 * 1024 * 1024 + 1)
    if len(body) > 20 * 1024 * 1024 or hashlib.sha256(body).hexdigest() != VIRTUALENV_SHA256:
        raise ValueError("Virtualenv archive checksum/size mismatch; bootstrap was not executed")
    atomic_write(archive, body)
    return archive


def ensure_environment(directory, cache_dir=DEFAULT_CACHE):
    """Дополняет частичное окружение pip, не удаляя пользовательские файлы."""
    directory = Path(directory).absolute()
    if directory.is_symlink() or directory.resolve() in {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}:
        raise ValueError("The environment must be a separate directory, not the system interpreter")
    python = environment_python(directory)
    check = ("import sys, pip; from pathlib import Path; "
             "assert sys.prefix != sys.base_prefix; "
             f"assert sys.version_info[:2] == {sys.version_info[:2]!r}; "
             "assert pip.__file__ is not None; "
             "assert Path(pip.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())")
    if python.exists():
        result = subprocess.run([str(python), "-I", "-c", check], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            return python
        print("Repairing incomplete environment:\n" + result.stdout, flush=True)
    archive = get_virtualenv(cache_dir)
    run_command([sys.executable, archive, "--python", sys.executable,
                 "--no-download", "--no-periodic-update", "--pip", "embed",
                 "--app-data", Path(cache_dir) / "app-data", directory])
    run_command([python, "-I", "-c", check])
    run_command([python, "-I", "-m", "pip", "--version"])
    return python


def requirements_fingerprint(path):
    """При смене Python требования проверяются вновь, даже если requirements не менялись."""
    return hashlib.sha256(Path(path).read_bytes() + sys.version.encode() + VIRTUALENV_VERSION.encode()).hexdigest()
