# SPDX-FileCopyrightText: 2019 Martin Fitzpatrick
# SPDX-FileCopyrightText: 2023 Joni Hyttinen <joni.hyttinen@uef.fi
#
# SPDX-License-Identifier: MIT

import sys
import traceback
import inspect

from PySide6.QtCore import QObject, Signal, QRunnable, Slot


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished:
        No data

    error:
        tuple (exctype, value, traceback.format_exc())

    result:
        object data returned from processing, anything
    """
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(float)


class Worker(QRunnable):
    """
    Class for creating worker threads that can emit progress signals.
    Additionally, the instances of the class emit finished, result,
    or error signals in response to runner function's behaviour.

    The runner function may at any point emit a progress signal. The
    progress signal is expected to pass an integer in 0-100 range. The runner
    function signature must include named argument "progress_callback".

    If the runner finishes successfully, a result signal is emitted.
    The result signal will pass the return value of the runner function,
    and may be an arbitrary object.

    If the runner function raises an exception, a traceback is printed,
    and an error signal is emitted. The exception data is passed in the
    emitted error signal.

    After the runner function has exited, regardless of success or failure,
    a finished signal is emitted. The finished signal will pass no data.
    """

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Inject progress_callback only if the target function accepts it
        sig = inspect.signature(self.fn)
        if "progress_callback" in sig.parameters:
            self.kwargs["progress_callback"] = self.signals.progress

    @Slot()
    def run(self):
        # Initialize the runner function with passed args and kwargs
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()