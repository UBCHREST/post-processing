import pathlib


def expand_path(path_pattern):
    p = pathlib.Path(path_pattern)
    return p.parent.expanduser().glob(p.name)
