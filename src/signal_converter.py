from collections import deque
from concurrent.futures import ThreadPoolExecutor
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Signal

from signal_constants import DEFAULT_SAMPLE_RATE

class SignalConverter:
    """Convert a CSV-signal file into a binary data file.

    This SignalConverter is intended to be run as a thread within an instance of the `Worker`-class.
    Usage example:
         # Get a thread pool
         threadpool = QThreadPool.globalInstance()

         # Create a SignalConverter instance
         converter = SignalConverter()

         # Prepare a Worker instance with `start`-method as the thread entry point
         worker = Worker(converter.start, source_signal_path=self.signal_path, target_signal_path=signal_path)

         # Connect worker's progress signal into a progress bar or dialog
         worker.signals.progress.connect(progress.setValue)

         # Start the thread
         threadpool.start(worker)
    """

    # Flag to indicate the conversion process should be cancelled.
    cancelled: bool = False
    # Raw CSV bytes parsed per worker chunk
    CHUNK_BYTES: int = 8 * 1024 * 1024

    def cancel(self):
        self.cancelled = True

    @staticmethod
    def _parse_chunk(raw_bytes: bytes) -> np.ndarray:
        """Parse a CSV byte chunk into a contiguous C-order int16 array (releases GIL in C)."""
        if not raw_bytes.strip():
            return np.empty((0, 2), dtype=np.int16)
        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            header=None,
            dtype=np.int16,
            engine="c",
            skip_blank_lines=True,
        )
        return np.ascontiguousarray(df.to_numpy(dtype=np.int16))

    def start(
        self,
        source_signal_path: Path,
        target_signal_path: Path,
        progress_callback: Signal | None = None,
    ) -> bool:
        self.cancelled = False
        source_signal_path = Path(source_signal_path)
        target_signal_path = Path(target_signal_path)

        # Make sure the progress is at 0 %
        if progress_callback:
            progress_callback.emit(0)
        if self.cancelled:
            return False

        total_bytes = source_signal_path.stat().st_size
        max_workers = os.cpu_count() or 1
        pending = deque()

        # Stream line-aligned chunks into a thread pool and write parsed int16 blocks in order
        with source_signal_path.open("rb") as src:
            header = src.readline()
            if not header:
                raise StopIteration

            bytes_done = len(header)
            with target_signal_path.open("wb") as dst, ThreadPoolExecutor(max_workers=max_workers) as pool:
                while True:
                    # Keep a bounded number of chunks in flight so memory stays small on huge files
                    while len(pending) < max_workers * 2 and not self.cancelled:
                        buf = src.read(self.CHUNK_BYTES)
                        if not buf:
                            break
                        if not buf.endswith(b"\n"):
                            buf += src.readline()
                        pending.append((len(buf), pool.submit(self._parse_chunk, buf)))

                    if not pending or self.cancelled:
                        break

                    chunk_len, future = pending.popleft()
                    arr = future.result()
                    if arr.size:
                        dst.write(arr.tobytes())

                    bytes_done += chunk_len
                    if progress_callback and total_bytes > 0 and bytes_done < total_bytes:
                        progress_callback.emit((bytes_done / total_bytes) * 100)

                if self.cancelled:
                    for _, fut in pending:
                        fut.cancel()

        if self.cancelled:
            if target_signal_path.exists():
                target_signal_path.unlink()
            return False

        if progress_callback:
            # Finish the progress bar.
            progress_callback.emit(100)

        if self.cancelled:
            if target_signal_path.exists():
                target_signal_path.unlink()
            if progress_callback:
                progress_callback.emit(0)
            return False

        return True