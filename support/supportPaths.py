import pathlib


def expand_path(path_pattern):
    p = pathlib.Path(path_pattern)
    return sorted(list(p.parent.expanduser().glob(p.name)))
