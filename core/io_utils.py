"""Page-cache hygiene for large checkpoint writes on Linux filesystems.

After each checkpoint write, flush files and ask the kernel to release their
clean cache pages. This keeps merge-family runs from accumulating one full
model's worth of page cache per task under memory-accounted job runners.
"""
from __future__ import annotations
import os


def drop_page_cache(path: str) -> None:
    """fsync + POSIX_FADV_DONTNEED every regular file under ``path``.

    ``fsync`` forces dirty pages out to the filesystem; ``fadvise`` then drops
    clean pages from the process's cache charge. Already-flushed files
    fsync in O(1), so calling this once per task on the whole seed output dir
    is cheap. Never raises: cache hygiene must not kill a training run.
    """
    if not hasattr(os, "posix_fadvise"):        # non-Linux: no-op
        return
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                fd = os.open(os.path.join(root, name), os.O_RDONLY)
            except OSError:
                continue                         # vanished/unreadable: skip
            try:
                os.fsync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
            finally:
                os.close(fd)
