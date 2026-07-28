#!/usr/bin/env python
"""Run any command in its own session so it outlives the launching shell.

Long fits started from an agent or CI shell get killed when that shell's process
group is torn down, and ``nohup ... &`` does not prevent it -- runs here have died
mid-epoch with no traceback and no non-zero exit, which is what that looks like.
``setsid`` would do the job but macOS does not ship it, so this is the double-fork
equivalent: fork, ``setsid`` to become a session leader, fork again so the process
can never reacquire a controlling terminal, then ``exec`` the real command.

``exec`` happens before the child imports anything, which matters: forking a
process that has already started torch's or BLAS's thread pools is unsafe, so a
generic launcher is actually safer here than a ``--detach`` flag inside a script
that has already imported them.

Usage::

    python examples/exploratory/detach.py --log runs/sweep.log -- \
        python examples/exploratory/baseline_capture.py run --tag foo --arms density
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="stdout+stderr destination")
    ap.add_argument("--pid", default=None, help="defaults to the log path with .pid")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- then the command")
    args = ap.parse_args()

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("no command given; put it after a bare --")

    log_path = Path(args.log)
    pid_path = Path(args.pid) if args.pid else log_path.with_suffix(".pid")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"detaching: log -> {log_path}")

    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    pid_path.write_text(f"{os.getpid()}\n")
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
    sys.exit(0)
