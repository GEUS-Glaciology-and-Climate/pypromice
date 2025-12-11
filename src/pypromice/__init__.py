from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("pypromice")
except PackageNotFoundError:
    __version__ = "unknown"
