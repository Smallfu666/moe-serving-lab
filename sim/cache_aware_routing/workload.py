"""Inputs shared by the validation replay and the routing simulation.

The replay file is the frozen output of `main-study/harness/load_trajectories.py` (sha256 recorded in each launch's
`input_sha.txt`). A turn's prompt is `seq[:prompt_len]`; the server's own output never reappears in a later prompt
(the replay sends the recorded answer), so output pages get keys no prompt can match.
"""

import hashlib
import json
import os
from pathlib import Path

from prefix_cache import page_keys

REPLAY_SHA256 = "baf4efba580d0664387aa182a75ad6e61d44919b772421a26b8f3d6697f1d7ff"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Sessions:
    """Page keys of each session's shared sequence, computed once per (session index, page size)."""

    def __init__(self, path, check_sha=True):
        if check_sha and sha256_file(path) != REPLAY_SHA256:
            raise SystemExit(f"{path}: sha256 differs from the replay file the runs used")
        with open(path) as f:
            self.lines = f.readlines()
        self.index = {json.loads(line[:120].split(',"seq"')[0] + "}")["session"]: i
                      for i, line in enumerate(self.lines)}
        self.ids = {i: s for s, i in self.index.items()}
        self._keys, self._turns = {}, {}

    def session_id(self, i):
        return self.ids[i]

    def turns(self, i):
        if i not in self._turns:
            self._load(i, None)
        return self._turns[i]

    def keys(self, i, page):
        if (i, page) not in self._keys:
            self._load(i, page)
        return self._keys[(i, page)]

    def _load(self, i, page):
        d = json.loads(self.lines[i])
        self._turns[i] = [(t["prompt_len"], t["answer_len"]) for t in d["turns"]]
        if page is not None:
            self._keys[(i, page)] = page_keys(d["seq"], page)


def request_keys(sess_keys, prompt_len, total_len, page, uid):
    """Pages SGLang keeps for a finished request: whole pages of prompt + output, unique past the prompt's pages."""
    shared = min(prompt_len // page, len(sess_keys))
    n = total_len // page
    return sess_keys[:shared] + [("u", uid, j) for j in range(n - shared)]


def launch_dir(raw, label):
    """Launch directory for a label such as "E3 R16-FP8 L1", from the private map `$LAUNCH_MAP` (label -> path in raw)."""
    return Path(raw) / json.loads(Path(os.environ["LAUNCH_MAP"]).read_text())[label]
