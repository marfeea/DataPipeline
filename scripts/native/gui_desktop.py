"""Verify the Windows desktop and this process's visible native windows."""

import ctypes
import os
from ctypes import wintypes


def desktop_evidence():
    user = ctypes.windll.user32
    user.GetThreadDesktop.restype = wintypes.HANDLE
    handle = user.GetThreadDesktop(ctypes.windll.kernel32.GetCurrentThreadId())
    buffer = ctypes.create_unicode_buffer(256)
    needed = wintypes.DWORD()
    if not user.GetUserObjectInformationW(wintypes.HANDLE(handle), 2, buffer, 512, ctypes.byref(needed)):
        raise RuntimeError("Cannot verify the desktop")
    result = dict(pid=os.getpid(), desktop=buffer.value, windows=[])
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def collect(hwnd, _):
        pid = wintypes.DWORD()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == os.getpid():
            title = ctypes.create_unicode_buffer(512)
            user.GetWindowTextW(hwnd, title, 512)
            rect = wintypes.RECT()
            user.GetWindowRect(hwnd, ctypes.byref(rect))
            result["windows"].append(dict(handle=int(hwnd), title=title.value,
                                          visible=bool(user.IsWindowVisible(hwnd)),
                                          minimized=bool(user.IsIconic(hwnd)),
                                          rect=[rect.left, rect.top, rect.right, rect.bottom]))
        return True

    user.EnumWindows(collect, 0)
    return result
