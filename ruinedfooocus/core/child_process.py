"""Keep native inference processes tied to the RuinedFooocus session."""

import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys


class WindowsJob:
    def __init__(self, memory_limit=None):
        from ctypes import wintypes as w

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("process_time", ctypes.c_int64),
                ("job_time", ctypes.c_int64),
                ("flags", w.DWORD),
                ("min_working_set", ctypes.c_size_t),
                ("max_working_set", ctypes.c_size_t),
                ("active_processes", w.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority", w.DWORD),
                ("scheduling", w.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("basic", BasicLimits),
                ("io_counters", ctypes.c_uint64 * 6),
                ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t),
                ("peak_job_memory", ctypes.c_size_t),
            ]

        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        self.api.CreateJobObjectW.restype = w.HANDLE
        self.api.SetInformationJobObject.argtypes = [
            w.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            w.DWORD,
        ]
        self.api.SetInformationJobObject.restype = w.BOOL
        self.api.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.api.AssignProcessToJobObject.restype = w.BOOL
        self.api.CloseHandle.argtypes = [w.HANDLE]
        self.api.CloseHandle.restype = w.BOOL
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if memory_limit is not None:
            limits.basic.flags |= 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
            limits.process_memory = memory_limit
        if not self.api.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def attach(self, process):
        if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def start_process(command, *, memory_limit=None, **kwargs):
    job = WindowsJob(memory_limit=memory_limit) if os.name == "nt" else None
    if sys.platform.startswith("linux"):
        # Arm the death signal in a fresh interpreter, not a threaded preexec_fn.
        command = [sys.executable, str(Path(__file__).resolve()), str(os.getpid()), *command]
    process = None
    try:
        process = subprocess.Popen(command, **kwargs)
        if job:
            job.attach(process)
        return process, job
    except BaseException:
        if process is not None:
            process.kill()
            process.wait()
        if job:
            job.close()
        raise


if __name__ == "__main__":
    parent = int(sys.argv[1])
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "Cannot attach inference process to its parent")
    if os.getppid() != parent:
        sys.exit(1)
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)
