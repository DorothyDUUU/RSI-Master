"""Per-run Claude Code permission policy (port of cc_branching, verbatim).

Generates a ``settings.json`` for each run that whitelists the public dirs
(``additionalDirectories``) and denies escapes.  Under ``--dangerously-skip-
permissions`` (the bypass mode this harness runs) allow rules and the
additionalDirectories whitelist stop binding — **deny rules are the only
permission-layer construct that still applies in every mode** — so this module
is defense in depth for the headless agent:

* relative ``../`` / ``../../`` traversal in the file tools and in shell
  commands,
* absolute-path reads/writes of sensitive OS roots (``//etc/**`` etc.),
* shell commands that *contain* a sensitive-root path anywhere (matches
  ``cat -n /etc/passwd``, ``grep -r x /etc``, ``cd /tmp && cat /etc/x`` …),
* the agent's own claude-invisible ``configs/`` system state (run.json /
  transcript — ground truth it must not read).

The hard boundary is still the process-level one: the agent's cwd is the
private workspace and anything outside it must be explicitly whitelisted.  A
determined agent can still leak out of these *string* rules via subprocesses
(``python -c "print(open('/etc/passwd').read())"``) or absolute paths under
the whitelisted prefix — that needs an OS sandbox (bwrap / container / non-root
user), not permission rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Never-legitimate absolute trees for a post-training agent.  Each becomes a
# ``//``-anchored Read+Edit deny (gitignore syntax, ``**`` crosses directories)
# and a Bash contain-anywhere deny ``Bash(*/etc/*)`` (``*`` matches any sequence
# incl. spaces, so the path token can sit anywhere in the command).  $SHARED_ROOT is
# deliberately excluded — the public_dirs live under it.
_OS_ROOTS = [
    "/etc", "/root", "/home", "/var", "/proc", "/sys",
    "/usr", "/bin", "/sbin", "/opt", "/boot", "/dev", "/run", "/srv",
]


def build_settings(config_dir: Path, public_dirs: list[str]) -> dict[str, Any]:
    """Return the settings dict for a run's Claude Code session.

    ``config_dir`` is the claude-INVISIBLE ``configs/`` dir (the workspace the
    agent sees is its sibling ``workspace/``).  Everything here is deny rules —
    the only permission layer that still binds under
    ``--dangerously-skip-permissions``.
    """
    # Note (claude 2.1.x): only Edit(path) and Read(path) rules match file
    # permission checks — Edit covers every file-editing tool, Read covers every
    # file-reading tool.  Write/Glob/LS forms are ignored by the checker.
    # Path rules use gitignore syntax: `//path` anchors at the filesystem root,
    # `../*` matches one parent segment only (use `../**` for the whole subtree).
    cfg = str(Path(config_dir).resolve())

    deny = [
        # Relative upward traversal above the workspace (one segment + subtree).
        "Read(../*)",
        "Read(../**)",
        "Edit(../*)",
        "Edit(../**)",
        "Read(../../*)",
        "Read(../../**)",
        "Edit(../../*)",
        "Edit(../../**)",
        # Shell escapes: cd/cat/ls/find up-and-out.
        "Bash(cd ../*)",
        "Bash(cat ../*)",
        "Bash(ls ../*)",
        "Bash(cd ../../*)",
        "Bash(cat ../../*)",
        "Bash(ls ../../*)",
        "Bash(find ../*)",
        "Bash(find ../../*)",
        # Bare `cd /` (exact match); `cd /etc/…` etc. are caught by the
        # contain-anywhere rules below.  (The old "Bash(cd /)&&*" was malformed
        # — the `&&*` sits outside the rule and is ignored.)
        "Bash(cd /)",
        # The agent's own system state: run.json/transcript are ground truth it
        # must not read.  Reachable only via absolute path under bypass, so deny
        # the exact config tree.  `//` anchor needs the path without its leading
        # slash (gitignore `//Users/alice/**` form).
        f"Read(//{cfg.lstrip('/')}/**)",
        f"Edit(//{cfg.lstrip('/')}/**)",
        f"Bash(*{cfg}*)",
        # Home reference in any shell command (`cat ~/.bashrc`, `cd ~/.ssh`).
        # Bash rules do NOT expand `~`, so match the literal `~/` token.
        "Bash(*~/*)",
        "Read(~/.ssh/**)",
        "Edit(~/.ssh/**)",
    ]

    # Absolute-path file-tool denies + shell contain-anywhere denies for the
    # sensitive OS roots.
    for root in _OS_ROOTS:
        deny.append(f"Read(//{root.lstrip('/')}/**)")
        deny.append(f"Edit(//{root.lstrip('/')}/**)")
        deny.append(f"Bash(*{root}/*)")

    return {
        "permissions": {
            "additionalDirectories": [str(d) for d in public_dirs],
            "deny": deny,
        }
    }


def write_settings(config_dir: Path, public_dirs: list[str]) -> Path:
    """Write ``<config_dir>/settings.json`` and return its path."""
    cfg = build_settings(config_dir, public_dirs)
    path = Path(config_dir) / "settings.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path
