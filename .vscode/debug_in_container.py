#!/usr/bin/env python3
"""Run a script under debugpy inside the MentorPi container and wait for VS Code to attach.

Used by the "debug-in-mentorpi-container" task in tasks.json.
"""
import runpy
import sys

import debugpy

script = sys.argv[1]
debugpy.listen(("0.0.0.0", 5678))
print("DEBUGPY_READY - waiting for VS Code to attach", flush=True)
debugpy.wait_for_client()

sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
