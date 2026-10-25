#!/usr/bin/env python3
"""Run a script under debugpy inside the MentorPi container and wait for VS Code to attach.

Used by the "debug-in-mentorpi-container" task in tasks.json.
"""
import os
import runpy
import sys
import threading

# Matched by the task's beginsPattern so VS Code knows the task has started.
print("DEBUGPY_STARTING", flush=True)

import debugpy

ATTACH_TIMEOUT = 60  # seconds; stops a leftover helper holding port 5678 if VS Code never attaches


def give_up():
    print(f"No debugger attached within {ATTACH_TIMEOUT}s - exiting", flush=True)
    os._exit(1)


script = sys.argv[1]
debugpy.listen(("0.0.0.0", 5678))
print("DEBUGPY_READY - waiting for VS Code to attach", flush=True)
timer = threading.Timer(ATTACH_TIMEOUT, give_up)
timer.daemon = True
timer.start()
debugpy.wait_for_client()
timer.cancel()

sys.argv = sys.argv[1:]
# Search the script's own folder first, like "python3 script.py" does (not .vscode/)
sys.path[0] = os.path.dirname(os.path.abspath(script))
runpy.run_path(script, run_name="__main__")
