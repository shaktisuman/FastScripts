#!/usr/bin/env python3
"""
nl2lldb - translate natural-language English into valid LLDB debugger commands.

Design goals
------------
* Correctness: every emitted command is a canonical LLDB command (not an alias),
  and its command/sub-command *path* is validated against LLDB itself.
* Coverage: a static command tree encodes the full standard LLDB command set
  (all top-level commands + sub-commands). When the real ``lldb`` module is
  importable, the validator queries LLDB's own command interpreter via the
  completion + ``help`` handlers, so even plugin-added commands are covered.
* Performance: all regexes/intent signals are compiled once at import time,
  scoring is a flat linear pass, and the command tree is cached. A single
  Translator instance can process tens of thousands of inputs per second.

Validation strategy (satisfies "use lldb command handlers to validate ... or use
its help handler to show possible suggestions for sub-commands"):
    * ``LLDBBackend`` uses ``SBCommandInterpreter.CommandExists`` and
      ``HandleCompletion`` to confirm a command path exists and to enumerate
      sub-commands, and ``HandleCommand('help <path>')`` to surface help.
    * ``StaticBackend`` provides the same interface from the bundled tree so the
      tool (and its tests) run anywhere, even without LLDB installed.

Usage
-----
    $ python3 nl2lldb.py "set a breakpoint at main"
    breakpoint set --name main

    # words may be given out of order (n-gram matching):
    $ python3 nl2lldb.py "main breakpoint set"
    breakpoint set --name main

    $ echo "step over" | python3 nl2lldb.py --json
    $ python3 nl2lldb.py --demo

After ``pip install .`` the same tool is available as the ``nl2lldb`` command.
See HOW_TO_USE.txt for the full guide.
"""

from __future__ import annotations

import argparse
import difflib
import functools
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

__version__ = "1.0.0"

__all__ = [
    "COMMAND_TREE", "ALIASES", "StaticBackend", "LLDBBackend", "get_backend",
    "Validator", "Extracted", "extract", "Intent", "INTENTS", "Translation",
    "Translator", "make_translator", "__version__",
]

# ---------------------------------------------------------------------------
# Static LLDB command tree (standard command set).  A node maps a command name
# to a dict of its sub-commands; a leaf is an empty dict.  Used by StaticBackend
# for validation/suggestions and as ground truth in tests.
# ---------------------------------------------------------------------------
def _leaves(*names: str) -> Dict[str, dict]:
    return {n: {} for n in names}

COMMAND_TREE: Dict[str, dict] = {
    "apropos": {},
    "breakpoint": {
        "clear": {},
        "command": _leaves("add", "delete", "list"),
        "delete": {},
        "disable": {},
        "enable": {},
        "list": {},
        "modify": {},
        "name": _leaves("add", "configure", "delete", "list"),
        "read": {},
        "set": {},
        "write": {},
    },
    "bugreport": _leaves("unwind"),
    "command": {
        "alias": {},
        "container": _leaves("add", "delete", "list"),
        "delete": {},
        "import": {},
        "regex": {},
        "script": _leaves("add", "clear", "delete", "import", "list"),
        "source": {},
        "unalias": {},
    },
    "disassemble": {},
    "expression": {},
    "frame": {
        "diagnose": {},
        "info": {},
        "recognizer": _leaves("add", "clear", "delete", "info", "list"),
        "select": {},
        "variable": {},
    },
    "gdb-remote": {},
    "gui": {},
    "help": {},
    "kdp-remote": {},
    "language": {
        "cplusplus": _leaves("demangle"),
        "objc": {
            "class-table": _leaves("dump"),
            "tagged-pointer": _leaves("info"),
        },
        "swift": {},
    },
    "log": {
        "disable": {},
        "dump": {},
        "enable": {},
        "list": {},
        "timers": _leaves("disable", "dump", "enable", "increment", "reset"),
    },
    "memory": {
        "find": {},
        "history": {},
        "read": {},
        "region": {},
        "tag": _leaves("read", "write"),
        "write": {},
    },
    "platform": {
        "connect": {},
        "disconnect": {},
        "file": _leaves("close", "open", "read", "write"),
        "get-file": {},
        "get-permissions": {},
        "get-size": {},
        "install": {},
        "list": {},
        "mkdir": {},
        "process": _leaves("attach", "info", "launch", "list"),
        "put-file": {},
        "select": {},
        "settings": {},
        "shell": {},
        "status": {},
        "target-install": {},
    },
    "plugin": _leaves("list", "load"),
    "process": {
        "attach": {},
        "connect": {},
        "continue": {},
        "detach": {},
        "handle": {},
        "interrupt": {},
        "kill": {},
        "launch": {},
        "load": {},
        "plugin": {"packet": _leaves("history", "send", "speed-test", "xfer-size")},
        "save-core": {},
        "signal": {},
        "status": {},
        "trace": _leaves("dump", "save", "start", "stop"),
        "unload": {},
    },
    "quit": {},
    "register": _leaves("info", "read", "write"),
    "script": {},
    "scripting": _leaves("run", "template"),
    "session": _leaves("history", "save"),
    "settings": {
        "append": {},
        "clear": {},
        "insert-after": {},
        "insert-before": {},
        "list": {},
        "read": {},
        "remove": {},
        "replace": {},
        "set": {},
        "show": {},
        "write": {},
    },
    "source": _leaves("info", "list"),
    "statistics": _leaves("disable", "dump", "enable"),
    "target": {
        "create": {},
        "delete": {},
        "dump": _leaves("typesystem"),
        "list": {},
        "modules": {
            "dump": _leaves(
                "line-table", "objfile", "pcm-info", "sections",
                "separate-debug-info", "symtab",
            ),
            "list": {},
            "load": {},
            "lookup": {},
            "search-paths": _leaves("add", "clear", "insert", "list", "query"),
            "show-unwind": {},
        },
        "select": {},
        "stop-hook": _leaves("add", "delete", "disable", "enable", "list"),
        "symbols": _leaves("add"),
        "variable": {},
    },
    "thread": {
        "backtrace": {},
        "continue": {},
        "exception": {},
        "info": {},
        "jump": {},
        "list": {},
        "plan": _leaves("discard", "list", "prune"),
        "return": {},
        "select": {},
        "siginfo": {},
        "step-in": {},
        "step-inst": {},
        "step-inst-over": {},
        "step-out": {},
        "step-over": {},
        "step-scripted": {},
        "until": {},
    },
    "type": {
        "category": _leaves("define", "delete", "disable", "enable", "list"),
        "filter": _leaves("add", "clear", "delete", "list"),
        "format": _leaves("add", "clear", "delete", "info", "list"),
        "lookup": {},
        "summary": _leaves("add", "clear", "delete", "info", "list"),
        "synthetic": _leaves("add", "clear", "delete", "list"),
    },
    "version": {},
    "watchpoint": {
        "command": _leaves("add", "delete", "list"),
        "delete": {},
        "disable": {},
        "enable": {},
        "ignore": {},
        "list": {},
        "modify": {},
        "set": _leaves("expression", "variable"),
    },
}

# Common single-token aliases -> canonical base command path (space separated).
ALIASES: Dict[str, str] = {
    "b": "breakpoint set", "br": "breakpoint", "tbreak": "breakpoint set",
    "bt": "thread backtrace",
    "c": "process continue", "cont": "process continue", "continue": "process continue",
    "r": "process launch", "run": "process launch",
    "n": "thread step-over", "next": "thread step-over",
    "s": "thread step-in", "step": "thread step-in",
    "so": "thread step-out", "finish": "thread step-out",
    "si": "thread step-inst", "ni": "thread step-inst-over",
    "p": "expression", "print": "expression", "call": "expression",
    "po": "expression", "e": "expression", "expr": "expression",
    "v": "frame variable", "var": "frame variable",
    "f": "frame select", "fr": "frame",
    "t": "thread",
    "x": "memory read",
    "reg": "register",
    "dis": "disassemble", "di": "disassemble", "disas": "disassemble",
    "image": "target modules", "file": "target create",
    "attach": "process attach", "detach": "process detach", "kill": "process kill",
    "q": "quit", "exit": "quit",
    "l": "source list", "list": "source list",
    "up": "frame select", "down": "frame select",
    "j": "thread jump", "jump": "thread jump",
}

# Keyword -> top-level command, for fallback suggestions on unrecognised input.
KEYWORD_TO_CMD: Dict[str, str] = {
    "breakpoint": "breakpoint", "break": "breakpoint", "bp": "breakpoint",
    "watchpoint": "watchpoint", "watch": "watchpoint",
    "memory": "memory", "mem": "memory",
    "register": "register", "reg": "register",
    "thread": "thread", "frame": "frame",
    "process": "process", "target": "target",
    "disassemble": "disassemble", "disassembly": "disassemble",
    "source": "source", "settings": "settings", "setting": "settings",
    "type": "type", "log": "log", "logging": "log",
    "expression": "expression", "expr": "expression",
    "platform": "platform", "command": "command",
}


# ---------------------------------------------------------------------------
# Validation backends
# ---------------------------------------------------------------------------
class StaticBackend:
    """Validation backend backed by the bundled :data:`COMMAND_TREE`.

    Command-path lookups are pure (the tree is immutable at runtime), so
    ``command_exists`` and ``subcommands`` are memoised with ``lru_cache``.
    Both return fresh copies, so callers can never mutate the cached values.
    """

    def __init__(self, tree: Dict[str, dict] = COMMAND_TREE):
        self.tree = tree
        # Per-instance caches keyed on the path tuple.  Per-instance (rather
        # than a module-level cache) keeps backends with different trees from
        # colliding, and lets each be cleared independently.
        self._exists = functools.lru_cache(maxsize=None)(self._exists_uncached)
        self._subs = functools.lru_cache(maxsize=None)(self._subs_uncached)

    def _node(self, path: Tuple[str, ...]) -> Optional[dict]:
        node = self.tree
        for p in path:
            if not isinstance(node, dict) or p not in node:
                return None
            node = node[p]
        return node

    def _exists_uncached(self, path: Tuple[str, ...]) -> bool:
        return self._node(path) is not None

    def _subs_uncached(self, path: Tuple[str, ...]) -> Tuple[str, ...]:
        node = self._node(path)
        return tuple(sorted(node.keys())) if isinstance(node, dict) else ()

    def command_exists(self, path: List[str]) -> bool:
        return self._exists(tuple(path))

    def subcommands(self, path: List[str]) -> List[str]:
        return list(self._subs(tuple(path)))

    def cache_clear(self) -> None:
        self._exists.cache_clear()
        self._subs.cache_clear()

    def help_text(self, path: List[str]) -> str:
        subs = self.subcommands(path)
        if subs:
            return "Available sub-commands of '%s': %s" % (" ".join(path), ", ".join(subs))
        if self.command_exists(path):
            return "'%s' takes arguments/options (no sub-commands)." % " ".join(path)
        return "Unknown command: %s" % " ".join(path)

    @property
    def name(self) -> str:
        return "static"


class LLDBBackend:
    """Validation backend backed by a live LLDB command interpreter.

    Uses LLDB's *own* handlers: ``CommandExists`` / ``HandleCompletion`` to test
    command paths and enumerate sub-commands, and ``HandleCommand('help ...')``
    to fetch help text.  Results are cached for speed.
    """

    def __init__(self, lldb_module):
        self.lldb = lldb_module
        self.debugger = lldb_module.SBDebugger.Create()
        self.debugger.SetAsync(False)
        self.interp = self.debugger.GetCommandInterpreter()
        # Completion queries hit the live interpreter, so memoise them.
        self._subs = functools.lru_cache(maxsize=None)(self._subs_uncached)

    def _complete(self, line: str) -> List[str]:
        matches = self.lldb.SBStringList()
        # (current_line, cursor_pos, match_start, max_return, matches)
        self.interp.HandleCompletion(line, len(line), 0, -1, matches)
        out = [matches.GetStringAtIndex(i) for i in range(matches.GetSize())]
        # Index 0 is the common-prefix to insert, not an actual match.
        return [m.strip() for m in out[1:] if m and m.strip()]

    def _subs_uncached(self, path: Tuple[str, ...]) -> Tuple[str, ...]:
        line = (" ".join(path) + " ") if path else ""
        return tuple(sorted({m for m in self._complete(line) if not m.startswith("-")}))

    def subcommands(self, path: List[str]) -> List[str]:
        return list(self._subs(tuple(path)))

    def command_exists(self, path: List[str]) -> bool:
        if not path:
            return False
        if len(path) == 1:
            try:
                if self.interp.CommandExists(path[0]):
                    return True
            except Exception:
                pass
            return path[0] in self.subcommands([])
        return path[-1] in self.subcommands(path[:-1])

    def cache_clear(self) -> None:
        self._subs.cache_clear()

    def help_text(self, path: List[str]) -> str:
        res = self.lldb.SBCommandReturnObject()
        self.interp.HandleCommand("help " + " ".join(path), res)
        return (res.GetOutput() or res.GetError() or "").strip()

    def shutdown(self) -> None:
        try:
            self.lldb.SBDebugger.Destroy(self.debugger)
        except Exception:
            pass

    @property
    def name(self) -> str:
        return "lldb"


def _import_lldb():
    """Best-effort import of the ``lldb`` python module (never raises)."""
    try:
        import lldb  # type: ignore
        return lldb
    except Exception:
        pass
    try:
        from shutil import which
        binary = which("lldb")
        if binary:
            out = subprocess.run([binary, "-P"], capture_output=True, text=True, timeout=10)
            path = out.stdout.strip()
            if path and os.path.isdir(path):
                sys.path.insert(0, path)
                import lldb  # type: ignore
                return lldb
    except Exception:
        pass
    return None


def get_backend(force_static: bool = False):
    """Return an LLDB-backed validation backend if possible, else static."""
    if not force_static:
        mod = _import_lldb()
        if mod is not None:
            try:
                return LLDBBackend(mod)
            except Exception:
                pass
    return StaticBackend()


# ---------------------------------------------------------------------------
# Validator: turns a candidate command string into (path, validity, suggestions)
# ---------------------------------------------------------------------------
class Validator:
    def __init__(self, backend=None, aliases: Dict[str, str] = ALIASES):
        self.backend = backend or StaticBackend()
        self.aliases = aliases

    def _expand_first(self, token: str) -> List[str]:
        exp = self.aliases.get(token.lower())
        return exp.split() if exp else [token]

    def command_path(self, command: str) -> Tuple[List[str], int]:
        """Return (validated command path, index of first non-path token)."""
        toks = command.split()
        if not toks:
            return [], 0
        path = self._expand_first(toks[0])
        consumed = 1
        while consumed < len(toks):
            cand = path + [toks[consumed]]
            if self.backend.command_exists(cand):
                path = cand
                consumed += 1
            else:
                break
        return path, consumed

    def analyze(self, command: str) -> dict:
        toks = command.split()
        path, consumed = self.command_path(command)
        valid = bool(path) and self.backend.command_exists(path)
        suggestions: List[str] = []
        note = ""
        if not valid:
            tops = self.backend.subcommands([]) or sorted(COMMAND_TREE.keys())
            first = toks[0] if toks else ""
            suggestions = difflib.get_close_matches(first, tops, n=6, cutoff=0.5)
            note = "Unrecognised command '%s'." % first if first else "Empty command."
        else:
            subs = self.backend.subcommands(path)
            rem = toks[consumed:]
            if subs and rem and not rem[0].startswith("-"):
                close = difflib.get_close_matches(rem[0], subs, n=6, cutoff=0.4)
                suggestions = close or subs
                note = "'%s' expects a sub-command (e.g. %s)." % (
                    " ".join(path), ", ".join(suggestions[:4]))
        return {
            "valid": valid,
            "path": path,
            "subcommands": self.backend.subcommands(path) if valid else [],
            "suggestions": suggestions,
            "note": note,
        }


# ---------------------------------------------------------------------------
# Natural-language entity extraction
# ---------------------------------------------------------------------------
_IDENT_CORE = r"[A-Za-z_][A-Za-z0-9_]*"
_RE_FUNC_OK = re.compile(r"^~?" + _IDENT_CORE + r"(?:::~?" + _IDENT_CORE + r")*$")
_RE_VAR_OK = re.compile(r"^" + _IDENT_CORE + r"(?:(?:->|\.)" + _IDENT_CORE + r"|\[\d+\])*$")
_RE_REG_OK = re.compile(r"^\$?[A-Za-z][A-Za-z0-9_]*$")

_STRIP = ".,;:!?()[]{}'\"`"

# Pre-compiled extraction patterns
RE_FILELINE = re.compile(r"([\w./+\-]+\.[A-Za-z][A-Za-z0-9]*)\s*:\s*(\d+)")
RE_LINE_OF_FILE = re.compile(r"\bline\s+(\d+)\s+(?:of|in)\s+(?:file\s+)?([\w./+\-]+\.[A-Za-z]\w*)", re.I)
RE_FILE_THEN_LINE = re.compile(r"\bfile\s+([\w./+\-]+\.[A-Za-z]\w*)\D{0,15}?\bline\s+(\d+)", re.I)
RE_HEX = re.compile(r"0[xX][0-9a-fA-F]+")
RE_LINE = re.compile(r"\bline\s+#?\s*(\d+)", re.I)
RE_PID = re.compile(r"\b(?:pid|process\s*id|process\s+number|process)\s+#?\s*(\d+)|\b(\d+)\s+(?:pid|process\s*id)\b", re.I)
RE_NUM = re.compile(r"-?\b\d+\b")
RE_COUNT_TIMES = re.compile(r"\b(\d+)\s+(?:times|hits?)\b", re.I)
RE_IGNORE = re.compile(r"\bignore(?:\s+count)?\s+(?:of\s+|it\s+|them\s+)?(\d+)", re.I)
RE_BYTES = re.compile(r"\b(\d+)\s+bytes?\b", re.I)
RE_COND = re.compile(r"\b(?:if|when(?:ever)?|condition|provided(?:\s+that)?|where)\s+(.+)$", re.I)
RE_PRINT = re.compile(
    r"\b(?:print(?:\s+out)?|evaluate|eval|compute|calculate|"
    r"what(?:'s|\s+is)\s+the\s+value\s+of|the\s+value\s+of|value\s+of|"
    r"show\s+(?:me\s+)?the\s+value\s+of)\s+(.+)$", re.I)
RE_PO = re.compile(
    r"\b(?:po|print\s+object|object\s+description\s+of|describe\s+object|describe)\s+(.+)$", re.I)
RE_ARGS = re.compile(
    r"\b(?:with\s+(?:args?|arguments?|parameters?)|passing|args?\s*[:=]|arguments?\s*[:=])\s+(.+)$", re.I)
RE_STOP_AT_ENTRY = re.compile(
    r"\b(?:stop|break|halt)\s+(?:at|on)\s+(?:the\s+)?entry(?:\s+point)?|at\s+the\s+start\b", re.I)
RE_ONE_SHOT = re.compile(
    r"\b(?:temporary|one[\s-]?shot|one[\s-]?time|single[\s-]?use|tbreak|just\s+once|only\s+once)\b", re.I)
RE_SETTING_SET = re.compile(
    r"\b(?:set(?:ting)?|change|configure)\s+(?:the\s+)?(?:setting\s+)?([\w][\w.\-]*\.[\w.\-]+)\s*"
    r"(?:to|=|as)?\s*(.+)$", re.I)
RE_SETTING_SHOW = re.compile(
    r"\b(?:show|get|display|print)\s+(?:the\s+)?setting\s+([\w][\w.\-]+)", re.I)
RE_HELP = re.compile(
    r"\b(?:help(?:\s+(?:with|on|for))?|how\s+(?:do\s+i|to)\s+use|usage\s+of|"
    r"man(?:ual)?\s+(?:for|of)|docs?\s+for|documentation\s+for)\s+(?:the\s+)?(?:command\s+)?(.+)$", re.I)
RE_APROPOS = re.compile(
    r"\b(?:apropos|search\s+(?:the\s+)?help\s+for|search\s+for\s+commands?\s+(?:about|for|related\s+to)?|"
    r"find\s+commands?\s+(?:about|for|related\s+to)|which\s+commands?\s+(?:deal\s+with|relate\s+to))\s+(.+)$", re.I)
RE_REG_WRITE = re.compile(
    r"\b(?:write|set|store|put)\s+(?:register\s+|to\s+register\s+)?(\$?[A-Za-z]\w*)\s+"
    r"(?:to|=|with(?:\s+value)?|equal\s+to)\s+(-?0[xX][0-9a-fA-F]+|-?\d+)", re.I)
RE_REG_WRITE2 = re.compile(
    r"\b(?:write|set|store|put)\s+(-?0[xX][0-9a-fA-F]+|-?\d+)\s+(?:to|into|in)\s+"
    r"(?:register\s+)?(\$?[A-Za-z]\w*)", re.I)
RE_MEM_WRITE = re.compile(
    r"\bwrite\s+(0[xX][0-9a-fA-F]+|-?\d+)\s+(?:to|at|into|in)\s+"
    r"(?:address\s+|memory\s+(?:at\s+)?|location\s+)?(0[xX][0-9a-fA-F]+)", re.I)
RE_RETURN_VAL = re.compile(
    r"\breturn(?:ing|s)?\s+(?:(?:a\s+)?value\s+(?:of\s+)?)?(-?0[xX][0-9a-fA-F]+|-?\d+)", re.I)
RE_REG_NAMED = re.compile(
    r"\b(?:register\s+(\$?[A-Za-z][A-Za-z0-9]*)|(\$?[A-Za-z][A-Za-z0-9]*)\s+register)\b", re.I)
RE_ALL = re.compile(r"\ball\b", re.I)

# Trigger / qualifier / stop word sets for name extraction
_FUNC_TRIGGERS = {"at", "in", "on", "to", "into", "inside", "function", "method",
                  "symbol", "routine", "break", "breakpoint", "reaches", "reach",
                  "hits", "enters", "disassemble"}
_FUNC_QUALIFIERS = {"the", "a", "an", "function", "method", "symbol", "routine", "named", "called",
                    "at", "in", "on", "to", "into", "inside"}
_FUNC_STOP = {"line", "address", "pid", "here", "there", "current", "entry", "start", "it",
              "execution", "if", "when", "whenever", "unless", "while"}

_VAR_TRIGGERS = {"variable", "var", "on", "of", "watch", "watchpoint", "value", "changes"}
_VAR_QUALIFIERS = {"the", "a", "an", "value", "of", "variable", "var", "local", "my", "on", "global"}
_VAR_STOP = {"line", "address", "screen", "stack", "it", "changes", "memory", "this", "frame"}

_REG_TRIGGERS = {"register", "reg"}
_REG_QUALIFIERS = {"the", "value", "of", "contents", "content"}
_REG_STOP: set = set()

_PROC_TRIGGERS = {"named", "called", "name"}
_PROC_QUALIFIERS = {"the", "a", "an", "process", "program", "app", "application", "executable"}
_PROC_STOP: set = set()


def _clean(tok: str) -> str:
    return tok.strip(_STRIP)


def _name_after(tokens: List[str], triggers: set, qualifiers: set, stop: set,
                valid: "re.Pattern", exclude: Optional[set] = None) -> Optional[str]:
    """Scan left-to-right; after a trigger word, skip qualifiers and return the
    first token that looks like a valid name (stopping at any *stop* word).
    Tokens in *exclude* (the known command vocabulary) are skipped over rather
    than returned, so scrambled input like 'breakpoint set main' yields 'main'
    instead of the command word 'set'."""
    n = len(tokens)
    for i in range(n):
        if tokens[i].lower().strip(_STRIP) in triggers:
            j = i + 1
            while j < n:
                w = _clean(tokens[j])
                lw = w.lower()
                if not w:
                    j += 1
                    continue
                if lw in qualifiers:
                    j += 1
                    continue
                if exclude and lw in exclude:
                    j += 1
                    continue
                if lw in stop:
                    break
                if valid.match(w):
                    return w
                break
    return None


def _dollar_register(tokens: List[str]) -> Optional[str]:
    for t in tokens:
        c = _clean(t)
        if re.match(r"^\$[A-Za-z][A-Za-z0-9_]*$", c):
            return c[1:]
    return None


def _strip_expr(s: str) -> str:
    s = s.strip().strip(_STRIP).strip()
    s = re.sub(r"^(?:the\s+|value\s+of\s+|variable\s+|var\s+|local\s+|global\s+)+", "", s, flags=re.I)
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1]
    return s.strip()


def _collect_command_words() -> set:
    """Every command/sub-command name in the tree, plus alias tokens."""
    words = set()

    def walk(node):
        for k, sub in node.items():
            for part in re.split(r"[\s\-]+", k):
                if part:
                    words.add(part.lower())
            walk(sub)
    walk(COMMAND_TREE)
    words.update(ALIASES.keys())
    words.update(KEYWORD_TO_CMD.keys())
    return words


# Vocabulary that should never be treated as a user-supplied identifier (used by
# the order-independent identifier finder).  It is the controlled command
# vocabulary plus the natural-language verbs/descriptors the parser understands.
_EXCLUDE_WORDS = _collect_command_words() | {
    # action verbs / synonyms
    "show", "display", "see", "view", "print", "dump", "examine", "inspect",
    "list", "get", "read", "write", "store", "put", "place", "create", "make",
    "insert", "add", "give", "attach", "set", "change", "configure", "modify",
    "delete", "remove", "clear", "unset", "drop", "enable", "disable", "turn",
    "activate", "deactivate", "switch", "select", "go", "move", "pick", "run",
    "launch", "start", "execute", "begin", "continue", "resume", "proceed",
    "keep", "going", "carry", "stop", "halt", "pause", "suspend", "interrupt",
    "kill", "terminate", "end", "detach", "disconnect", "finish", "step",
    "stepping", "next", "into", "out", "over", "back", "evaluate", "eval",
    "compute", "calculate", "describe", "look", "lookup", "find", "search",
    "watch", "ignore", "skip", "open", "quit", "exit", "close", "leave",
    "reaches", "reach", "hits", "hit", "enters", "enter", "dive",
    # descriptors / nouns the grammar consumes
    "breakpoint", "breakpoints", "watchpoint", "watchpoints", "thread",
    "threads", "frame", "frames", "process", "program", "app", "application",
    "executable", "binary", "target", "memory", "register", "registers",
    "source", "code", "lines", "line", "assembly", "machine", "instruction",
    "instructions", "address", "addresses", "bytes", "byte", "object",
    "description", "value", "values", "variable", "variables", "var", "vars",
    "local", "locals", "global", "module", "modules", "image", "images",
    "library", "libraries", "shared", "dylib", "dylibs", "symbol", "setting",
    "settings", "version", "gui", "help", "apropos", "backtrace", "stack",
    "call", "trace", "condition", "conditional", "pid", "id", "number",
    "count", "times", "time", "temporary", "permanent", "entry", "point",
    "current", "this", "selected", "previous", "last", "first", "topmost",
    "args", "arguments", "argument", "parameters", "parameter", "passing",
    "status", "state", "running", "info", "information", "about", "rid",
    "all", "here", "there", "now", "early", "immediately", "single", "one",
    "where", "what", "which", "how", "use", "usage", "command", "commands",
    "pc", "frames", "func", "function", "method", "routine",
}

# Minimal connective/stop words for the identifier finder.
_ID_STOP = {
    "a", "an", "the", "to", "of", "on", "in", "at", "for", "with", "and", "or",
    "me", "i", "is", "it", "my", "from", "if", "when", "that", "this", "please",
    "just", "some", "as", "by", "into", "up", "down",
}

_RE_PLAIN_IDENT = re.compile(r"^~?[A-Za-z_][A-Za-z0-9_]*(?:::~?[A-Za-z_][A-Za-z0-9_]*)*$")


def _loose_identifier(tokens: List[str]) -> Optional[str]:
    """Order-independent identifier finder: return the first token that looks
    like a user symbol/variable and is NOT part of the known command grammar.
    Lets scrambled input like 'main breakpoint set' still yield 'main'."""
    for t in tokens:
        c = t.strip(_STRIP)
        if not c:
            continue
        low = c.lower()
        if low in _EXCLUDE_WORDS or low in _ID_STOP:
            continue
        if any(ch.isdigit() for ch in c) or ":" in low or "." in c or "/" in c:
            continue  # skip numbers, file:line, addresses, paths
        if c.startswith("-") or c.startswith("$"):
            continue
        if _RE_PLAIN_IDENT.match(c):
            return c
    return None


@dataclass
class Extracted:
    text: str
    low: str
    tokens: List[str]
    numbers: List[int] = field(default_factory=list)
    hex_addrs: List[str] = field(default_factory=list)
    file: Optional[str] = None
    line: Optional[int] = None
    func: Optional[str] = None
    var: Optional[str] = None
    reg: Optional[str] = None
    pid: Optional[int] = None
    proc_name: Optional[str] = None
    condition: Optional[str] = None
    expression: Optional[str] = None
    po_expr: Optional[str] = None
    args: Optional[str] = None
    one_shot: bool = False
    stop_at_entry: bool = False
    count: Optional[int] = None
    byte_count: Optional[int] = None
    has_all: bool = False
    setting_name: Optional[str] = None
    setting_value: Optional[str] = None
    help_cmd: Optional[str] = None
    apropos_term: Optional[str] = None


def extract(text: str) -> Extracted:
    text = text.strip()
    low = text.lower()
    tokens = text.split()
    ex = Extracted(text=text, low=low, tokens=tokens)
    if not text:
        return ex

    ex.numbers = [int(m) for m in RE_NUM.findall(text)]
    ex.hex_addrs = RE_HEX.findall(text)
    ex.has_all = bool(RE_ALL.search(low))

    m = RE_FILELINE.search(text) or RE_LINE_OF_FILE.search(text) or RE_FILE_THEN_LINE.search(text)
    if m:
        if m.re is RE_LINE_OF_FILE:
            ex.line, ex.file = int(m.group(1)), m.group(2)
        else:
            ex.file, ex.line = m.group(1), int(m.group(2))
    if ex.line is None:
        ml = RE_LINE.search(text)
        if ml:
            ex.line = int(ml.group(1))

    mp = RE_PID.search(text)
    if mp:
        ex.pid = int(mp.group(1) or mp.group(2))

    mb = RE_BYTES.search(text)
    if mb:
        ex.byte_count = int(mb.group(1))

    mc = RE_COUNT_TIMES.search(text) or RE_IGNORE.search(text)
    if mc:
        ex.count = int(mc.group(1))

    ex.func = _name_after(tokens, _FUNC_TRIGGERS, _FUNC_QUALIFIERS, _FUNC_STOP, _RE_FUNC_OK, _EXCLUDE_WORDS)
    ex.var = _name_after(tokens, _VAR_TRIGGERS, _VAR_QUALIFIERS, _VAR_STOP, _RE_VAR_OK, _EXCLUDE_WORDS)
    ex.reg = _dollar_register(tokens) or _name_after(
        tokens, _REG_TRIGGERS, _REG_QUALIFIERS, _REG_STOP, _RE_REG_OK, _EXCLUDE_WORDS)
    if ex.reg and ex.reg.startswith("$"):
        ex.reg = ex.reg[1:]
    if not ex.reg:
        mrn = RE_REG_NAMED.search(text)
        if mrn:
            cand = (mrn.group(1) or mrn.group(2) or "").lstrip("$")
            if cand.lower() not in _EXCLUDE_WORDS and cand.lower() not in {
                    "the", "all", "value", "values", "contents", "content",
                    "this", "that", "a", "an"}:
                ex.reg = cand
    ex.proc_name = _name_after(tokens, _PROC_TRIGGERS, _PROC_QUALIFIERS, _PROC_STOP, _RE_FUNC_OK, _EXCLUDE_WORDS)

    # Order-independent fallback: if the trigger-based scan found no name but a
    # lone unknown identifier is present (e.g. scrambled "main breakpoint set"
    # or "rax read register"), use it for the empty name/variable/register slot.
    if ex.func is None or ex.var is None or ex.reg is None:
        loose = _loose_identifier(tokens)
        if loose is not None:
            if ex.func is None:
                ex.func = loose
            if ex.var is None:
                ex.var = loose
            if ex.reg is None:
                ex.reg = loose

    mcond = RE_COND.search(text)
    if mcond:
        ex.condition = mcond.group(1).strip().strip(_STRIP).strip()

    mpo = RE_PO.search(text)
    if mpo:
        ex.po_expr = _strip_expr(mpo.group(1))
    mpr = RE_PRINT.search(text)
    if mpr:
        ex.expression = _strip_expr(mpr.group(1))

    margs = RE_ARGS.search(text)
    if margs:
        ex.args = margs.group(1).strip().strip(".").strip()

    ex.one_shot = bool(RE_ONE_SHOT.search(text))
    ex.stop_at_entry = bool(RE_STOP_AT_ENTRY.search(text))

    msh = RE_SETTING_SHOW.search(text)
    if msh:
        ex.setting_name = msh.group(1)
    else:
        mset = RE_SETTING_SET.search(text)
        if mset:
            ex.setting_name = mset.group(1)
            ex.setting_value = mset.group(2).strip().strip(_STRIP).strip()

    mh = RE_HELP.search(text)
    if mh:
        words = re.findall(r"[A-Za-z][\w\-]*", mh.group(1))
        ex.help_cmd = " ".join(words[:3]) if words else None

    ma = RE_APROPOS.search(text)
    if ma:
        ex.apropos_term = ma.group(1).strip().strip(_STRIP).strip()

    return ex


# ---------------------------------------------------------------------------
# Intent definitions
# ---------------------------------------------------------------------------
def _sig(pattern: str, weight: int) -> Tuple["re.Pattern", int]:
    return (re.compile(pattern, re.I), weight)


RenderResult = Tuple[Optional[str], List[str]]

# --- order-independent (n-gram / bag-of-words) matching -------------------
# The signal regexes above are order-sensitive (e.g. "step over").  To also
# accept the same words in any order ("over step", "main breakpoint set"), we
# derive an order-independent "bag" for each intent from its signal patterns:
# each signal contributes the set of literal words it contains, with the same
# weight.  At scoring time we award (weight * fraction-of-words-present) so a
# fully-present word set scores like the original signal and a partial set
# scores proportionally.  This is combined additively with the ordered score.

# Words that never help disambiguate; dropped from bags and from the token set.
_NG_STOP = {
    "a", "an", "the", "of", "to", "and", "or", "me", "i", "is", "it", "s",
    "do", "be", "by", "as", "that", "this", "please", "just", "some", "my",
    "for", "with", "am", "are", "was", "from", "if", "when", "whenever",
    "value", "values",
}
# Words that must NOT have a trailing 's' stripped (would corrupt them).
_STEM_KEEP = {
    "process", "status", "address", "class", "bus", "gui", "os", "this",
    "plus", "cross", "loss", "pass", "always", "less", "as", "is", "its",
}


def _stem(word: str) -> str:
    """Very small singulariser so plurals match their singular form."""
    w = word.lower()
    if len(w) > 3 and w.endswith("s") and w not in _STEM_KEEP:
        w = w[:-1]
    return w


def _ngram_tokens(tokens: List[str]) -> set:
    """Normalise raw tokens into a stemmed set for order-independent matching."""
    out = set()
    for t in tokens:
        c = t.strip(".,;:!?()[]{}'\"`").lower()
        if not c or c in _NG_STOP:
            continue
        out.add(_stem(c))
    return out


# --- regex -> order-independent "slots" compiler ---------------------------
# Each signal regex is compiled into a list of slots.  A slot is a list of
# alternative word-sets (OR), and a slot may be optional.  An input token set
# "matches" a signal iff every non-optional slot has at least one alternative
# whose words are all present.  Sequence => AND across slots; ``(?:a|b)`` => OR
# within a slot; trailing ``?`` => optional; ``\s*`` between letters lets the
# two words also match when written joined (watch point / watchpoint).

def _split_top_level(text: str, sep: str = "|") -> List[str]:
    parts, depth, cur = [], 0, []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            cur.append(text[i:i + 2]); i += 2; continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur)); cur = []
        else:
            cur.append(ch)
        i += 1
    parts.append("".join(cur))
    return parts


def _expand_slots(slots: List[Tuple[List[frozenset], bool]]) -> List[frozenset]:
    """Expand a slot list into the list of word-sets (OR) it can match."""
    combos = [frozenset()]
    for alts, optional in slots:
        choices = list(alts)
        if optional:
            choices = choices + [frozenset()]
        if not choices:
            continue
        new = []
        for base in combos:
            for ch in choices:
                new.append(base | ch)
                if len(new) > 64:
                    break
        combos = new
    return [c for c in combos if c]


def _seq_to_slots(seq: str) -> List[Tuple[List[frozenset], bool]]:
    """Parse a regex sub-sequence (no top-level '|') into ordered slots."""
    slots: List[Tuple[List[frozenset], bool]] = []
    i, n = 0, len(seq)
    prev_literal = False          # previous emitted element was a literal word
    joinable = False              # the separator just seen allows zero spaces
    chain: List[str] = []         # raw words in the current joinable chain
    chain_idx = -1                # index in `slots` of the current chain slot

    def reset_chain():
        nonlocal prev_literal, chain, chain_idx
        prev_literal = False
        chain = []
        chain_idx = -1

    while i < n:
        ch = seq[i]
        if ch == "\\":
            esc = seq[i + 1] if i + 1 < n else ""
            i += 2
            # \b is a boundary (keeps a chain joinable); \s is a separator.
            if esc in "sS":
                # look at following quantifier to decide join-ability
                q = seq[i] if i < n else ""
                joinable = (q == "*")
                if q in "*+?":
                    i += 1
                # \s+ or \s separates words (not joinable unless \s*)
                if not joinable:
                    prev_literal = prev_literal  # keep; sep only
            elif esc == "b":
                pass  # boundary: leave joinable/prev as-is
            else:
                reset_chain(); joinable = False  # \d \w etc: wildcard
            continue
        if ch == "(" and seq[i:i + 3] == "(?:":
            depth, j = 0, i
            while j < n:
                if seq[j] == "\\":
                    j += 2; continue
                if seq[j] == "(":
                    depth += 1
                elif seq[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            body = seq[i + 3:j]
            i = j + 1
            optional = False
            if i < n and seq[i] in "?*":
                optional = True; i += 1
            elif i < n and seq[i] == "+":
                i += 1
            alts: List[frozenset] = []
            for piece in _split_top_level(body, "|"):
                for ws in _expand_slots(_seq_to_slots(piece)):
                    alts.append(ws)
            alts = [a for a in alts if a]
            if alts:
                slots.append((alts, optional))
            reset_chain(); joinable = False
            continue
        if ch == "[":
            j = i + 1
            while j < n and seq[j] != "]":
                if seq[j] == "\\":
                    j += 2
                else:
                    j += 1
            i = j + 1
            if i < n and seq[i] in "*+?":
                i += 1
            reset_chain(); joinable = False  # char class: wildcard separator
            continue
        if ch.isalpha():
            k = i
            while k < n and seq[k].isalpha():
                k += 1
            word = seq[i:k]
            i = k
            optional = False
            if i < n and seq[i] == "?":
                # trailing ? makes the LAST letter optional, not the word
                i += 1
            elif i < n and seq[i] in "*+":
                i += 1
            if len(word) < 2 or word.lower() in _NG_STOP:
                reset_chain()
                continue
            if prev_literal and joinable and chain_idx >= 0:
                chain.append(word)
                and_set = frozenset(_stem(w) for w in chain)
                joined = frozenset([_stem("".join(chain))])
                slots[chain_idx] = ([and_set, joined], slots[chain_idx][1])
            else:
                chain = [word]
                slots.append(([frozenset([_stem(word)])], False))
                chain_idx = len(slots) - 1
            prev_literal = True
            joinable = False
            continue
        # any other character (anchors, quantifiers, etc.)
        if ch in "^$":
            pass
        else:
            reset_chain(); joinable = False
        i += 1
    return slots


def _compile_signal(pattern: str) -> List[Tuple[List[frozenset], bool]]:
    """Compile a full signal pattern (which may contain top-level '|')."""
    pieces = _split_top_level(pattern, "|")
    if len(pieces) == 1:
        return _seq_to_slots(pattern)
    # Top-level alternation: the whole signal is one OR slot.
    alts: List[frozenset] = []
    for piece in pieces:
        for ws in _expand_slots(_seq_to_slots(piece)):
            alts.append(ws)
    alts = [a for a in alts if a]
    return [(alts, False)] if alts else []


@dataclass
class Intent:
    name: str
    signals: List[Tuple["re.Pattern", int]]
    command: Optional[str] = None
    render: Optional[Callable[[Extracted], RenderResult]] = None
    priority: int = 0
    # Each entry: (compiled slots, weight).  Built from the signal patterns.
    groups: List[Tuple[list, int]] = field(default_factory=list)
    kw: Dict[str, int] = field(default_factory=dict)

    def build_bag(self) -> None:
        """Compile this intent's signals into order-independent slot groups and
        record the set of words involved (for idf)."""
        groups = []
        kw: Dict[str, int] = {}
        for rx, w in self.signals:
            slots = _compile_signal(rx.pattern)
            if not slots:
                continue
            groups.append((slots, w))
            for alts, _opt in slots:
                for alt in alts:
                    for word in alt:
                        if w > kw.get(word, 0):
                            kw[word] = w
        self.groups = groups
        self.kw = kw

    def score(self, low: str) -> int:
        """Ordered score: sum of weights of signals whose regex matches."""
        return sum(w for rx, w in self.signals if rx.search(low))

    def ngram_score(self, tok_set: set) -> float:
        """Order-independent score.  Each group whose every non-optional slot
        has an alternative fully present in the token set contributes its signal
        weight scaled by the discriminativeness (idf) of the matched words; the
        contributions are summed."""
        idf = _IDF
        total = 0.0
        for slots, w in self.groups:
            matched_words = []
            ok = True
            for alts, optional in slots:
                hit = None
                for alt in alts:
                    if alt <= tok_set:
                        hit = alt
                        break
                if hit is None:
                    if not optional:
                        ok = False
                        break
                else:
                    matched_words.extend(hit)
            if ok and matched_words:
                gidf = max(idf.get(t, 1.0) for t in matched_words)
                total += w * gidf
        return total


# Inverse-document-frequency over intent keywords, filled once the intent table
# is built (see below).  A word in few intents is highly discriminative.
_IDF: Dict[str, float] = {}


def _build_idf(intents: List["Intent"]) -> Dict[str, float]:
    df: Dict[str, int] = {}
    for it in intents:
        for t in it.kw:
            df[t] = df.get(t, 0) + 1
    return {t: 1.0 / d for t, d in df.items()}


def _q(s: str) -> str:
    """Quote a condition/value for the LLDB command line if needed."""
    if not s:
        return "''"
    if not any(c.isspace() for c in s) and "'" not in s and '"' not in s:
        return s
    if "'" not in s:
        return "'%s'" % s
    if '"' not in s:
        return '"%s"' % s
    return "'" + s.replace("'", "'\\''") + "'"


# ---- render helpers --------------------------------------------------------
def r_bp_set(ex: Extracted) -> RenderResult:
    notes: List[str] = []
    parts = ["breakpoint set"]
    if ex.file and ex.line is not None:
        parts.append("--file %s --line %d" % (ex.file, ex.line))
    elif ex.func:
        parts.append("--name %s" % ex.func)
    elif ex.hex_addrs:
        parts.append("--address %s" % ex.hex_addrs[0])
    elif ex.line is not None:
        parts.append("--line %d" % ex.line)
        notes.append("No file given; LLDB will use the current source file.")
    else:
        notes.append("Could not find a location (function, file:line, or address).")
    if ex.condition:
        parts.append("--condition %s" % _q(ex.condition))
    if ex.one_shot:
        parts.append("--one-shot true")
    return " ".join(parts), notes


def r_bp_delete(ex: Extracted) -> RenderResult:
    if ex.has_all or not ex.numbers:
        return "breakpoint delete", ["Deletes all breakpoints."]
    return "breakpoint delete %d" % ex.numbers[0], []


def _bp_simple(verb: str):
    def render(ex: Extracted) -> RenderResult:
        if ex.numbers and not ex.has_all:
            return "breakpoint %s %d" % (verb, ex.numbers[0]), []
        return "breakpoint %s" % verb, (["Applies to all breakpoints."] if not ex.numbers else [])
    return render


def r_bp_modify_cond(ex: Extracted) -> RenderResult:
    if not ex.condition:
        return "breakpoint modify", ["No condition found after 'if'/'when'."]
    bp = (" %d" % ex.numbers[0]) if ex.numbers else ""
    note = [] if ex.numbers else ["No breakpoint id given; applies to the last set breakpoint."]
    return "breakpoint modify --condition %s%s" % (_q(ex.condition), bp), note


def r_bp_ignore(ex: Extracted) -> RenderResult:
    nums = [n for n in ex.numbers if n != ex.count]
    bp_id = nums[0] if nums else (ex.numbers[0] if ex.numbers else None)
    count = ex.count if ex.count is not None else 1
    if bp_id is None:
        return "breakpoint modify --ignore-count %d" % count, ["No breakpoint id found."]
    return "breakpoint modify --ignore-count %d %d" % (count, bp_id), []


def r_launch(ex: Extracted) -> RenderResult:
    parts = ["process launch"]
    if ex.stop_at_entry:
        parts.append("--stop-at-entry")
    if ex.args:
        parts.append("-- %s" % ex.args)
    return " ".join(parts), []


def r_backtrace(ex: Extracted) -> RenderResult:
    if re.search(r"all\s+thread", ex.low):
        return "thread backtrace --all", []
    if ex.numbers:
        return "thread backtrace --count %d" % ex.numbers[0], []
    return "thread backtrace", []


def r_thread_select(ex: Extracted) -> RenderResult:
    if ex.numbers:
        return "thread select %d" % ex.numbers[0], []
    return "thread select", ["No thread number given."]


def r_frame_select(ex: Extracted) -> RenderResult:
    if ex.numbers:
        return "frame select %d" % ex.numbers[0], []
    return "frame select", ["No frame number given."]


def r_frame_variable(ex: Extracted) -> RenderResult:
    if ex.var:
        return "frame variable %s" % ex.var, []
    return "frame variable", []


def r_expression(ex: Extracted) -> RenderResult:
    expr = ex.expression or ex.po_expr or _loose_identifier(ex.tokens)
    if not expr:
        return "expression", ["No expression found to evaluate."]
    return "expression -- %s" % expr, []


def r_po(ex: Extracted) -> RenderResult:
    expr = ex.po_expr or ex.expression or _loose_identifier(ex.tokens)
    if not expr:
        return "expression -O --", ["No object expression found."]
    return "expression -O -- %s" % expr, []


def r_register_read(ex: Extracted) -> RenderResult:
    if ex.reg:
        return "register read %s" % ex.reg, []
    if re.search(r"\ball\b", ex.low):
        return "register read --all", []
    return "register read", []


def r_register_write(ex: Extracted) -> RenderResult:
    m = RE_REG_WRITE.search(ex.text)
    if m:
        return "register write %s %s" % (m.group(1).lstrip("$"), m.group(2)), []
    m2 = RE_REG_WRITE2.search(ex.text)
    if m2:
        return "register write %s %s" % (m2.group(2).lstrip("$"), m2.group(1)), []
    if ex.reg and ex.numbers:
        return "register write %s %d" % (ex.reg, ex.numbers[-1]), []
    return "register write", ["Need a register and a value (e.g. 'set register rax to 5')."]


def r_memory_read(ex: Extracted) -> RenderResult:
    if not ex.hex_addrs:
        return "memory read", ["No address found (e.g. 'read memory at 0x1000')."]
    parts = ["memory read"]
    if ex.byte_count:
        parts.append("--count %d" % ex.byte_count)
    parts.append(ex.hex_addrs[0])
    return " ".join(parts), []


def r_memory_write(ex: Extracted) -> RenderResult:
    m = RE_MEM_WRITE.search(ex.text)
    if m:
        return "memory write %s %s" % (m.group(2), m.group(1)), []
    if len(ex.hex_addrs) >= 1 and ex.numbers:
        return "memory write %s %d" % (ex.hex_addrs[0], ex.numbers[-1]), []
    return "memory write", ["Need an address and a value."]


def r_watchpoint_set(ex: Extracted) -> RenderResult:
    if ex.var:
        return "watchpoint set variable %s" % ex.var, []
    if ex.hex_addrs:
        return "watchpoint set expression -- %s" % ex.hex_addrs[0], []
    if ex.expression:
        return "watchpoint set expression -- %s" % ex.expression, []
    return "watchpoint set variable", ["Need a variable or address to watch."]


def r_watchpoint_delete(ex: Extracted) -> RenderResult:
    if ex.numbers and not ex.has_all:
        return "watchpoint delete %d" % ex.numbers[0], []
    return "watchpoint delete", (["Deletes all watchpoints."] if not ex.numbers else [])


def r_disassemble(ex: Extracted) -> RenderResult:
    if ex.func:
        return "disassemble --name %s" % ex.func, []
    if re.search(r"\b(?:current|this|selected)\s+frame\b", ex.low):
        return "disassemble --frame", []
    if re.search(r"\b(?:pc|program\s+counter)\b", ex.low):
        return "disassemble --pc", []
    if ex.hex_addrs:
        return "disassemble --start-address %s" % ex.hex_addrs[0], []
    return "disassemble", []


def r_source_list(ex: Extracted) -> RenderResult:
    parts = ["source list"]
    if ex.file:
        parts.append("--file %s" % ex.file)
    if ex.line is not None:
        parts.append("--line %d" % ex.line)
    return " ".join(parts), []


def r_attach(ex: Extracted) -> RenderResult:
    if ex.pid:
        return "process attach --pid %d" % ex.pid, []
    if ex.proc_name:
        return "process attach --name %s" % ex.proc_name, []
    return "process attach", ["Need a pid or process name."]


def r_settings_set(ex: Extracted) -> RenderResult:
    if ex.setting_name and ex.setting_value:
        return "settings set %s %s" % (ex.setting_name, ex.setting_value), []
    if ex.setting_name:
        return "settings set %s" % ex.setting_name, ["No value given."]
    return "settings set", ["Need a setting name (a dotted path) and a value."]


def r_settings_show(ex: Extracted) -> RenderResult:
    if ex.setting_name:
        return "settings show %s" % ex.setting_name, []
    return "settings show", []


def r_help(ex: Extracted) -> RenderResult:
    if ex.help_cmd:
        return "help %s" % ex.help_cmd, []
    return "help", []


def r_apropos(ex: Extracted) -> RenderResult:
    if ex.apropos_term:
        return "apropos %s" % ex.apropos_term, []
    return "apropos", ["Need a search term."]


def r_image_lookup(ex: Extracted) -> RenderResult:
    if ex.func:
        return "target modules lookup --name %s" % ex.func, []
    if ex.hex_addrs:
        return "target modules lookup --address %s" % ex.hex_addrs[0], []
    return "target modules lookup", ["Need a symbol name or address."]


def r_thread_until(ex: Extracted) -> RenderResult:
    if ex.line is not None:
        return "thread until %d" % ex.line, []
    if ex.numbers:
        return "thread until %d" % ex.numbers[0], []
    return "thread until", ["Need a line number."]


def r_thread_return(ex: Extracted) -> RenderResult:
    m = RE_RETURN_VAL.search(ex.text)
    if m:
        return "thread return %s" % m.group(1), []
    return "thread return", []


# ---- the intent table ------------------------------------------------------
def _build_intents() -> List[Intent]:
    bp = r"break\s*point|breakpoint|\bbreak\b|\bbp\b"
    return [
        # ---- breakpoints ----
        Intent("breakpoint_set", [
            _sig(r"\bset\s+(?:a\s+|the\s+|up\s+a\s+)?(?:\w+\s+){0,3}?(?:break\s*point|breakpoint)", 12),
            _sig(r"\b(?:add|put|place|create|make|insert)\s+(?:a\s+|the\s+)?(?:\w+\s+){0,3}?(?:break\s*point|breakpoint)", 11),
            _sig(r"\btbreak\b", 12),
            _sig(r"\bbreak\s+(?:at|on|in|when|if)\b", 9),
            _sig(r"\bstop\s+(?:at|on|in)\b", 7),
            _sig(r"\bstop\s+(?:execution\s+)?when\b.*\breach", 8),
            _sig(r"\bbreakpoint\b", 2), _sig(r"\bbreak\b", 1),
        ], render=r_bp_set),
        Intent("breakpoint_list", [
            _sig(r"\b(?:list|show|display|see|view)\b[^.]*\b(?:%s)" % bp, 11),
            _sig(r"\b(?:%s)\b[^.]*\b(?:list|status)\b" % bp, 10),
            _sig(r"\bwhat\s+breakpoints\b", 10),
            _sig(r"\bbreakpoint\b", 2),
        ], command="breakpoint list"),
        Intent("breakpoint_delete", [
            _sig(r"\b(?:delete|remove|clear|unset|drop|get\s+rid\s+of)\b[^.]*\b(?:%s)" % bp, 12),
            _sig(r"\bbreakpoint\b", 2),
        ], render=r_bp_delete),
        Intent("breakpoint_enable", [
            _sig(r"\b(?:re-?enable|enable|turn\s+on|activate|switch\s+on)\b[^.]*\b(?:%s)" % bp, 12),
            _sig(r"\bbreakpoint\b", 2),
        ], render=_bp_simple("enable")),
        Intent("breakpoint_disable", [
            _sig(r"\b(?:disable|turn\s+off|deactivate|switch\s+off)\b[^.]*\b(?:%s)" % bp, 12),
            _sig(r"\bbreakpoint\b", 2),
        ], render=_bp_simple("disable")),
        Intent("breakpoint_modify_condition", [
            _sig(r"\bmodify\b[^.]*\b(?:%s)" % bp, 12),
            _sig(r"\b(?:add|give|attach|set)\s+(?:a\s+)?condition\b", 10),
            _sig(r"\bmake\b[^.]*\bconditional\b", 9),
        ], render=r_bp_modify_cond),
        Intent("breakpoint_ignore", [
            _sig(r"\bignore\b[^.]*\b(?:%s)" % bp, 12),
            _sig(r"\bskip\b[^.]*\b(?:%s)" % bp, 9),
        ], render=r_bp_ignore),

        # ---- process / run control ----
        Intent("process_launch", [
            _sig(r"\blaunch\b", 11),
            _sig(r"\b(?:run|start|execute|begin)\b[^.]*\b(?:program|process|executable|app|target|binary|it|debugging)\b", 11),
            _sig(r"\bstart\s+running\b", 10),
            _sig(r"\brun\s+the\s+(?:program|app|binary|executable)\b", 11),
            _sig(r"^\s*run\b", 8), _sig(r"\brun\b", 4),
        ], render=r_launch),
        Intent("process_continue", [
            _sig(r"\bcontinue\b", 11), _sig(r"\bresume\b", 10),
            _sig(r"\bkeep\s+going\b", 10), _sig(r"\bcarry\s+on\b", 9),
            _sig(r"\bproceed\b", 8), _sig(r"\blet\s+it\s+run\b", 9),
            _sig(r"\bunpause\b", 8),
        ], command="process continue"),
        Intent("process_interrupt", [
            _sig(r"\binterrupt\b", 11),
            _sig(r"\b(?:pause|halt|suspend)\s+(?:the\s+)?(?:process|program|execution|target)\b", 10),
            _sig(r"\bhalt\s+execution\b", 9), _sig(r"\bpause\b", 5),
        ], command="process interrupt"),
        Intent("process_kill", [
            _sig(r"\bkill\b", 11),
            _sig(r"\b(?:terminate|end|stop)\s+(?:the\s+)?(?:process|program|target|debuggee)\b", 10),
            _sig(r"\bhalt\s+(?:the\s+)?(?:process|program)\b", 8),
        ], command="process kill"),
        Intent("process_detach", [
            _sig(r"\bdetach\b", 11),
            _sig(r"\bdisconnect\s+from\s+(?:the\s+)?process\b", 9),
        ], command="process detach"),
        Intent("process_status", [
            _sig(r"\bprocess\s+status\b", 12), _sig(r"\bprocess\s+state\b", 10),
            _sig(r"\bis\s+(?:it|the\s+(?:program|process))\s+running\b", 10),
            _sig(r"\b(?:program|process)\s+state\b", 9),
            _sig(r"\bstatus\s+of\s+the\s+process\b", 10),
        ], command="process status"),

        # ---- stepping ----
        Intent("thread_step_inst_over", [
            _sig(r"\bnext\s+(?:machine\s+)?instruction\b", 16),
            _sig(r"\bstep\s+over\s+(?:one\s+|a\s+|the\s+)?instruction\b", 16),
            _sig(r"\binstruction\s+(?:level\s+)?over\b", 9), _sig(r"\bni\b", 8),
        ], command="thread step-inst-over"),
        Intent("thread_step_inst", [
            _sig(r"\bstep\s+(?:one\s+|a\s+|the\s+|into\s+)?(?:machine\s+)?instruction\b", 11),
            _sig(r"\bsingle[\s-]?step\b", 10), _sig(r"\binstruction\s+step\b", 10),
            _sig(r"\bstep\s+by\s+(?:one\s+)?instruction\b", 11),
            _sig(r"\binstruction\b", 7), _sig(r"\bsi\b", 8),
        ], command="thread step-inst"),
        Intent("thread_step_over", [
            _sig(r"\bstep\s+over\b", 12), _sig(r"\bstep-over\b", 12),
            _sig(r"\bnext\s+line\b", 11), _sig(r"\bnext\s+statement\b", 10),
            _sig(r"\bgo\s+over\b", 8), _sig(r"\bexecute\s+(?:the\s+)?next\s+line\b", 9),
            _sig(r"\bstep\s+past\b", 8), _sig(r"\bnext\b", 7),
        ], command="thread step-over"),
        Intent("thread_step_in", [
            _sig(r"\bstep\s+in(?:to)?\b", 12), _sig(r"\bstep-in\b", 12),
            _sig(r"\benter\s+(?:the\s+)?(?:function|call|method)\b", 9),
            _sig(r"\bdive\s+into\b", 8), _sig(r"\bstep\b", 4),
        ], command="thread step-in"),
        Intent("thread_step_out", [
            _sig(r"\bstep\s+out(?:\s+of)?\b", 12), _sig(r"\bstep-out\b", 12),
            _sig(r"\bfinish\b", 11), _sig(r"\breturn\s+from\s+(?:the\s+)?(?:current\s+)?function\b", 10),
            _sig(r"\bstep\s+return\b", 9), _sig(r"\bget\s+out\s+of\s+(?:this\s+)?function\b", 9),
        ], command="thread step-out"),
        Intent("thread_until", [
            _sig(r"\b(?:run|continue|step|go)\s+(?:up\s+)?(?:to|until)\s+line\b", 11),
            _sig(r"\buntil\s+line\b", 11),
        ], render=r_thread_until),
        Intent("thread_return", [
            _sig(r"\bforce\s+return\b", 11), _sig(r"\breturn\s+immediately\b", 10),
            _sig(r"\breturn\s+(?:now|early)\b", 9), _sig(r"\breturn\s+(?:a\s+)?value\b", 9),
        ], render=r_thread_return),

        # ---- threads / frames ----
        Intent("thread_backtrace", [
            _sig(r"\bback\s*trace\b", 12), _sig(r"\bbt\b", 9),
            _sig(r"\b(?:call\s*stack|stack\s*trace|stack\s*frames?)\b", 10),
            _sig(r"\bwhere\b", 7),
            _sig(r"\b(?:show|print|display|dump|get)\b[^.]*\bstack\b", 8),
        ], render=r_backtrace),
        Intent("thread_list", [
            _sig(r"\b(?:list|show|display|see)\b[^.]*\bthreads?\b", 11),
            _sig(r"\ball\s+threads\b", 9), _sig(r"\bthreads?\s+list\b", 11),
        ], command="thread list"),
        Intent("thread_info", [
            _sig(r"\bthread\s+info\b", 11), _sig(r"\bcurrent\s+thread\b", 9),
            _sig(r"\binfo\s+(?:about|on)\s+(?:the\s+)?(?:current\s+)?thread\b", 10),
            _sig(r"\bthread\s+status\b", 9),
        ], command="thread info"),
        Intent("thread_select", [
            _sig(r"\b(?:select|switch\s+to|go\s+to|change\s+to|pick)\s+thread\b", 11),
            _sig(r"\bthread\s+select\b", 11),
        ], render=r_thread_select),
        Intent("frame_select", [
            _sig(r"\b(?:select|switch\s+to|go\s+to|change\s+to|pick)\s+frame\b", 12),
            _sig(r"\bframe\s+select\b", 12), _sig(r"\bframe\s+#?\d+\b", 8),
        ], render=r_frame_select),
        Intent("frame_up", [
            _sig(r"\b(?:go\s+up|move\s+up|one\s+frame\s+up|frame\s+up)\b", 11),
            _sig(r"\b(?:caller|parent)\s+frame\b", 9), _sig(r"^\s*up\b", 9),
        ], command="frame select --relative 1"),
        Intent("frame_down", [
            _sig(r"\b(?:go\s+down|move\s+down|one\s+frame\s+down|frame\s+down)\b", 11),
            _sig(r"\b(?:callee|child)\s+frame\b", 9), _sig(r"^\s*down\b", 9),
        ], command="frame select --relative -1"),
        Intent("frame_info", [
            _sig(r"\bframe\s+info\b", 11), _sig(r"\bcurrent\s+frame\b", 9),
            _sig(r"\bwhere\s+am\s+i\b", 11), _sig(r"\bcurrent\s+(?:location|position)\b", 9),
            _sig(r"\bwhat\s+(?:line|function)\s+am\s+i\s+(?:on|in)\b", 10),
        ], command="frame info"),
        Intent("frame_variable", [
            _sig(r"\blocal\s+variables?\b", 12), _sig(r"\b(?:frame|all)\s+variables?\b", 11),
            _sig(r"\b(?:show|list|print|dump|display)\s+(?:all\s+)?(?:the\s+)?(?:local\s+)?vari?(?:able)?s?\b", 10),
            _sig(r"\b(?:show|list|print|display)\s+locals\b", 11),
            _sig(r"\bvariables?\s+in\s+(?:this|the|current)\s+(?:frame|scope)\b", 10),
            _sig(r"\bvariable\b", 3), _sig(r"\bvar\b", 3),
        ], render=r_frame_variable),

        # ---- expressions / printing ----
        Intent("expression_po", [
            _sig(r"\bpo\b", 10), _sig(r"\bprint\s+object\b", 12),
            _sig(r"\bobject\s+description\b", 11), _sig(r"\bdescribe\s+object\b", 11),
            _sig(r"\bdescription\s+of\b", 7),
        ], render=r_po),
        Intent("expression", [
            _sig(r"\bprint\b", 7), _sig(r"\bevaluate\b", 9), _sig(r"\beval\b", 8),
            _sig(r"\b(?:compute|calculate)\b", 8),
            _sig(r"\bvalue\s+of\b", 7), _sig(r"\bwhat(?:'s|\s+is)\s+the\s+value\b", 9),
            _sig(r"\bexpression\b", 9), _sig(r"\bexpr\b", 9), _sig(r"\bcall\b", 6),
        ], render=r_expression, priority=-1),

        # ---- registers / memory ----
        Intent("register_write", [
            _sig(r"\b(?:write|set|store|put|change|modify)\b[^.]*\bregister\b", 11),
            _sig(r"\bregister\b[^.]*\b(?:to|=|with)\b", 9),
            _sig(r"\bset\s+\$?[a-z][a-z0-9]*\s+(?:to|=)\b", 8),
        ], render=r_register_write),
        Intent("register_read", [
            _sig(r"\b(?:read|show|print|display|dump|get|view|examine|inspect)\b[^.]*\bregisters?\b", 11),
            _sig(r"\bregisters?\b", 4), _sig(r"\bregister\s+(?:read|values?|contents?)\b", 11),
            _sig(r"\bwhat(?:'s| is)\s+in\s+\$?[a-z]+\b", 8),
        ], render=r_register_read),
        Intent("memory_write", [
            _sig(r"\bwrite\b[^.]*\b(?:memory|address|location)\b", 11),
            _sig(r"\bwrite\s+0[xX][0-9a-fA-F]+\b", 9),
            _sig(r"\bstore\b[^.]*\b(?:at\s+address|in\s+memory)\b", 9),
        ], render=r_memory_write),
        Intent("memory_read", [
            _sig(r"\b(?:read|show|print|display|dump|examine|inspect|view|look\s+at)\b[^.]*\b(?:memory|address|bytes)\b", 11),
            _sig(r"\bmemory\s+(?:at|read|contents?)\b", 11),
            _sig(r"\bexamine\s+0[xX]", 9), _sig(r"\bhex\s*dump\b", 9),
        ], render=r_memory_read),

        # ---- watchpoints ----
        Intent("watchpoint_list", [
            _sig(r"\b(?:list|show|display|see)\b[^.]*\bwatch\s*points?\b", 12),
            _sig(r"\bwatch\s*point\s+list\b", 11),
        ], command="watchpoint list"),
        Intent("watchpoint_delete", [
            _sig(r"\b(?:delete|remove|clear|drop)\b[^.]*\bwatch\s*points?\b", 12),
        ], render=r_watchpoint_delete),
        Intent("watchpoint_set", [
            _sig(r"\b(?:set|add|put|create|place)\b[^.]*\bwatch\s*point\b", 12),
            _sig(r"\bwatch\s+(?:the\s+)?(?:variable|var|value\s+of|when)\b", 11),
            _sig(r"\bbreak\s+(?:when|on)\b[^.]*\bchanges?\b", 10),
            _sig(r"\bwatch\b", 4),
        ], render=r_watchpoint_set),

        # ---- disassembly / source ----
        Intent("disassemble", [
            _sig(r"(?<![\w.\-])disassemble(?![\w.\-])", 12),
            _sig(r"(?<![\w.\-])disassembly(?![\w.\-])", 11),
            _sig(r"(?<![\w.\-])disas(?:m)?(?![\w.\-])", 10),
            _sig(r"\bshow\s+(?:the\s+)?assembly\b", 10),
            _sig(r"\bshow\s+(?:the\s+)?machine\s+code\b", 9),
        ], render=r_disassemble),
        Intent("source_list", [
            _sig(r"\b(?:list|show|display)\s+(?:the\s+)?(?:source|code|lines?)\b", 11),
            _sig(r"\bsource\s+(?:list|code)\b", 11), _sig(r"\bsee\s+the\s+source\b", 10),
            _sig(r"\bwhat(?:'s| is)\s+(?:the\s+)?(?:source|code)\b", 9),
        ], render=r_source_list),
        Intent("source_info", [
            _sig(r"\bsource\s+info\b", 11), _sig(r"\binfo\s+about\s+(?:the\s+)?source\b", 9),
        ], command="source info"),

        # ---- attach / images ----
        Intent("process_attach", [
            _sig(r"\battach\b[^.]*\b(?:process|pid|program|app)\b", 12),
            _sig(r"\battach\b", 9), _sig(r"\bconnect\s+to\s+(?:the\s+)?(?:process|pid)\b", 9),
        ], render=r_attach),
        Intent("image_lookup", [
            _sig(r"\b(?:image\s+lookup|look\s*up\s+(?:the\s+)?(?:symbol|function|address)|lookup\s+symbol)\b", 12),
            _sig(r"\bfind\s+(?:the\s+)?symbol\b", 10), _sig(r"\bsymbol\s+(?:info|lookup)\b", 10),
        ], render=r_image_lookup),
        Intent("image_list", [
            _sig(r"\b(?:list|show|display)\b[^.]*\b(?:modules?|images?|libraries|shared\s+libraries|dylibs?|so\s+files)\b", 12),
            _sig(r"\bimage\s+list\b", 11), _sig(r"\bloaded\s+(?:modules?|images?|libraries)\b", 10),
        ], command="target modules list"),

        # ---- settings / misc ----
        Intent("settings_show", [
            _sig(r"\b(?:show|list|display|get)\s+(?:the\s+)?(?:current\s+)?settings?\b", 13),
            _sig(r"\bsettings?\s+(?:show|list)\b", 11), _sig(r"\bcurrent\s+settings\b", 9),
        ], render=r_settings_show),
        Intent("settings_set", [
            _sig(r"\b(?:set|change|configure)\b[^.]*\bsetting\b", 13),
            _sig(r"\bsettings?\s+set\b", 11),
            _sig(r"\bset\s+[\w.\-]+\.[\w.\-]+\s+(?:to|=)\b", 9),
        ], render=r_settings_set),
        Intent("help", [
            _sig(r"^\s*help\b", 11), _sig(r"\bshow\s+help\b", 10),
            _sig(r"\b(?:list|show)\s+(?:all\s+)?commands\b", 10),
            _sig(r"\bhow\s+(?:do\s+i|to)\s+use\b", 10), _sig(r"\bhelp\s+(?:with|on|for)\b", 11),
            _sig(r"\b(?:man(?:ual)?|docs?|documentation)\s+(?:for|of)\b", 9), _sig(r"\bhelp\b", 6),
        ], render=r_help),
        Intent("apropos", [
            _sig(r"\bapropos\b", 12), _sig(r"\bsearch\s+(?:the\s+)?help\b", 18),
            _sig(r"\bfind\s+commands?\b", 10), _sig(r"\bsearch\s+for\s+commands?\b", 11),
            _sig(r"\bwhich\s+commands?\b", 9),
        ], render=r_apropos),
        Intent("version", [
            _sig(r"\bversion\b", 11), _sig(r"\bwhat\s+version\b", 11),
            _sig(r"\blldb\s+version\b", 12),
        ], command="version"),
        Intent("gui", [
            _sig(r"^\s*gui\b", 12), _sig(r"\b(?:open|show|launch|start)\s+(?:the\s+)?gui\b", 11),
            _sig(r"\b(?:terminal\s+)?ui\s+mode\b", 9),
        ], command="gui"),
        Intent("quit", [
            _sig(r"^\s*quit\b", 12), _sig(r"^\s*exit\b", 11),
            _sig(r"\b(?:quit|exit|close|leave)\s+(?:lldb|the\s+debugger|the\s+session|debugging)\b", 12),
            _sig(r"\bquit\b", 7), _sig(r"\bexit\b", 6),
        ], command="quit"),
    ]


INTENTS: List[Intent] = _build_intents()
for _it in INTENTS:
    _it.build_bag()
_IDF = _build_idf(INTENTS)


# ---------------------------------------------------------------------------
# Completeness rules: which intents need an argument we must extract, and the
# usage hints to suggest when that argument is missing.  Each entry maps an
# intent name to (predicate, suggestion_hints); predicate(ex) is True when the
# rendered command is complete.  Intents not listed are always complete (their
# defaults -- e.g. "register read" reads all, "breakpoint delete" deletes all
# -- are valid on their own).
# ---------------------------------------------------------------------------
def _has_reg_write(ex: "Extracted") -> bool:
    return bool(RE_REG_WRITE.search(ex.text) or RE_REG_WRITE2.search(ex.text)
                or (ex.reg and ex.numbers))


def _has_mem_write(ex: "Extracted") -> bool:
    return bool(RE_MEM_WRITE.search(ex.text) or (ex.hex_addrs and ex.numbers))


def _has_bp_ignore_id(ex: "Extracted") -> bool:
    ids = [n for n in ex.numbers if n != ex.count]
    return bool(ids) or (bool(ex.numbers) and ex.count is None)


INCOMPLETE_RULES = {
    "breakpoint_set": (
        lambda ex: bool(ex.func or ex.hex_addrs or ex.line is not None),
        ["breakpoint set --name <function>",
         "breakpoint set --file <file> --line <n>",
         "breakpoint set --address <0xADDRESS>"]),
    "breakpoint_modify_condition": (
        lambda ex: bool(ex.condition),
        ['breakpoint modify --condition "<expr>" <breakpoint-id>']),
    "breakpoint_ignore": (
        _has_bp_ignore_id,
        ["breakpoint modify --ignore-count <count> <breakpoint-id>"]),
    "thread_select": (
        lambda ex: bool(ex.numbers),
        ["thread select <thread-index>", "thread list  (to see thread ids)"]),
    "frame_select": (
        lambda ex: bool(ex.numbers),
        ["frame select <frame-index>"]),
    "thread_until": (
        lambda ex: ex.line is not None or bool(ex.numbers),
        ["thread until <line-number>"]),
    "expression": (
        lambda ex: bool(ex.expression or ex.po_expr),
        ["expression -- <expression>"]),
    "expression_po": (
        lambda ex: bool(ex.po_expr or ex.expression),
        ["expression -O -- <object-expression>"]),
    "register_write": (
        _has_reg_write,
        ["register write <register> <value>"]),
    "memory_read": (
        lambda ex: bool(ex.hex_addrs),
        ["memory read <0xADDRESS>", "memory read --count <n> <0xADDRESS>"]),
    "memory_write": (
        _has_mem_write,
        ["memory write <0xADDRESS> <value>"]),
    "watchpoint_set": (
        lambda ex: bool(ex.var or ex.hex_addrs or ex.expression),
        ["watchpoint set variable <variable>",
         "watchpoint set expression -- <expression>"]),
    "process_attach": (
        lambda ex: bool(ex.pid or ex.proc_name),
        ["process attach --pid <pid>", "process attach --name <process-name>"]),
    "settings_set": (
        lambda ex: bool(ex.setting_name and ex.setting_value),
        ["settings set <setting.name> <value>"]),
    "image_lookup": (
        lambda ex: bool(ex.func or ex.hex_addrs),
        ["target modules lookup --name <symbol>",
         "target modules lookup --address <0xADDRESS>"]),
    "apropos": (
        lambda ex: bool(ex.apropos_term),
        ["apropos <search-term>"]),
}

# Filler words ignored when deciding whether the input names only a command group.
_GROUP_FILLER = {"the", "a", "an", "please", "just", "some"}


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Translation:
    input: str
    command: Optional[str]
    intent: Optional[str]
    confidence: float
    valid: bool
    suggestions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    complete: bool = True

    def to_dict(self) -> dict:
        return {
            "input": self.input, "command": self.command, "intent": self.intent,
            "confidence": self.confidence, "valid": self.valid,
            "complete": self.complete,
            "suggestions": self.suggestions, "notes": self.notes,
        }

    def __str__(self) -> str:
        sug = (" Suggestions: " + ", ".join(self.suggestions)) if self.suggestions else ""
        if self.command:
            if self.complete:
                return self.command
            return "%s  # incomplete -- needs more info.%s" % (self.command, sug)
        return "# could not translate.%s" % sug


class Translator:
    """Translate natural-language strings into validated LLDB commands."""

    def __init__(self, validator: Optional[Validator] = None,
                 intents: List[Intent] = INTENTS, low_threshold: int = 4,
                 cache_size: Optional[int] = 4096, ngram_weight: float = 0.5):
        self.validator = validator or Validator()
        self.intents = intents
        self.low_threshold = low_threshold
        # Weight of the order-independent (n-gram) score relative to the ordered
        # score.  For fully out-of-order input the ordered score is ~0, so the
        # n-gram ranking decides; for in-order input the ordered score dominates.
        self.ngram_weight = ngram_weight
        # text -> Translation is deterministic for a given translator, so cache
        # it.  cache_size=0/None disables caching (each call recomputes).
        if cache_size:
            self._cached = functools.lru_cache(maxsize=cache_size)(self._translate_uncached)
        else:
            self._cached = self._translate_uncached

    # An ordered score at/above this means a real multi-word phrase matched in
    # order; we trust it and skip the order-independent tie-break.
    STRONG_ORDERED = 8.0

    def _best_intent(self, low: str, tok_set: set) -> Tuple[Optional[Intent], float]:
        """Rank intents.  If a strong, order-sensitive phrase matched we trust
        it (preserving precise behaviour).  Otherwise -- e.g. the words are out
        of order or only weak signals fired -- rank by ordered + ngram_weight *
        order-independent score so scrambled input still resolves correctly."""
        scored = []
        for it in self.intents:
            ordered = it.score(low)
            ngram = it.ngram_score(tok_set)
            scored.append((it, ordered, ordered + self.ngram_weight * ngram))
        if not scored:
            return None, 0.0

        ordered_best = max(scored, key=lambda x: (x[1], x[0].priority))
        if ordered_best[1] >= self.STRONG_ORDERED:
            return ordered_best[0], ordered_best[2]

        combined_best = max(scored, key=lambda x: (x[2], x[0].priority))
        if combined_best[2] <= 0:
            return None, 0.0
        return combined_best[0], combined_best[2]

    def _fallback(self, ex: Extracted) -> Tuple[List[str], str]:
        for tok in ex.tokens:
            cmd = KEYWORD_TO_CMD.get(tok.lower().strip(_STRIP))
            if cmd:
                subs = self.validator.backend.subcommands([cmd])
                note = ("Did not recognise the action. '%s' supports: %s. Try 'help %s'."
                        % (cmd, ", ".join(subs) if subs else "(options only)", cmd))
                return subs, note
        tops = self._top_commands()
        return tops[:12], "Could not interpret the request. Available top-level commands include the above."

    def _top_commands(self) -> List[str]:
        return self.validator.backend.subcommands([]) or sorted(COMMAND_TREE.keys())

    def _suggestions_for(self, command: Optional[str], ex: Extracted) -> List[str]:
        """Best-effort suggestions for when a complete, valid command could not
        be formed.  Prefer the sub-commands of whatever command group the output
        points at; otherwise fall back to keyword/top-level guidance."""
        if command:
            toks = command.split()
            if toks:
                path, _ = self.validator.command_path(command)
                base = path or self.validator._expand_first(toks[0])
                if base and self.validator.backend.command_exists(base):
                    subs = self.validator.backend.subcommands(base)
                    if subs:
                        return ["%s %s" % (" ".join(base), s) for s in subs][:12]
                close = difflib.get_close_matches(
                    toks[0].lower(), self._top_commands(), n=6, cutoff=0.4)
                if close:
                    return close
        sug, _ = self._fallback(ex)
        return sug

    def _command_group_only(self, ex: Extracted):
        """If the input names only a command group (e.g. just 'breakpoint' or
        'thread'), return (base_command, subcommands) so we can suggest the
        sub-commands instead of guessing an action."""
        words = [w.lower().strip(_STRIP) for w in ex.tokens]
        words = [w for w in words if w and w not in _GROUP_FILLER]
        if len(words) != 1:
            return None
        w = words[0]
        base = KEYWORD_TO_CMD.get(w) or KEYWORD_TO_CMD.get(w.rstrip("s"))
        if not base:
            return None
        subs = self.validator.backend.subcommands([base])
        if not subs:
            return None
        return base, subs

    def translate(self, text: str) -> Translation:
        """Translate one natural-language request (served from cache if seen)."""
        return self._cached(text)

    def cache_info(self):
        """Return lru_cache statistics for the translation cache, or None."""
        info = getattr(self._cached, "cache_info", None)
        return info() if info else None

    def cache_clear(self) -> None:
        """Clear the translation cache (and the backend lookup cache)."""
        clear = getattr(self._cached, "cache_clear", None)
        if clear:
            clear()
        backend_clear = getattr(self.validator.backend, "cache_clear", None)
        if backend_clear:
            backend_clear()

    def _translate_uncached(self, text: str) -> Translation:
        ex = extract(text)
        if not ex.text:
            return Translation(text, None, None, 0.0, False,
                               self._top_commands()[:12],
                               ["Empty input. Available top-level commands include the above."],
                               complete=False)

        # Partial input that only names a command group -> suggest sub-commands.
        group = self._command_group_only(ex)
        if group is not None:
            base, subs = group
            suggestions = ["%s %s" % (base, s) for s in subs]
            note = ("'%s' names a command group rather than a complete action; "
                    "choose a sub-command:" % ex.text)
            return Translation(text, None, None, 0.0, False, suggestions,
                               [note], complete=False)

        intent, score = self._best_intent(ex.low, _ngram_tokens(ex.tokens))
        if intent is None or score <= 0:
            sug, note = self._fallback(ex)
            return Translation(text, None, None, 0.0, False, sug, [note], complete=False)

        if intent.render is not None:
            command, notes = intent.render(ex)
        else:
            command, notes = intent.command, []

        confidence = round(min(1.0, score / 12.0), 2)
        if score < self.low_threshold:
            notes = notes + ["Low confidence (%d); please verify the command." % score]

        # Completeness: does this intent need an argument we could not extract?
        complete = True
        hint_suggestions: List[str] = []
        rule = INCOMPLETE_RULES.get(intent.name)
        if rule is not None and not rule[0](ex):
            complete = False
            hint_suggestions = list(rule[1])

        info = self.validator.analyze(command) if command else {
            "valid": False, "suggestions": [], "note": "", "subcommands": []}
        suggestions = hint_suggestions + list(info.get("suggestions") or [])
        if info.get("note"):
            notes = notes + [info["note"]]

        valid = bool(info.get("valid"))
        # Guarantee: if we could not produce a valid AND complete command, the
        # caller always gets actionable suggestions to choose from.
        if not (command and valid and complete) and not suggestions:
            suggestions = self._suggestions_for(command, ex)
            if not notes:
                notes = ["Could not form a complete, valid command; "
                         "did you mean one of the suggestions?"]

        return Translation(text, command, intent.name, confidence,
                           valid, suggestions, notes, complete=complete)

    def translate_many(self, texts: List[str]) -> List[Translation]:
        return [self.translate(t) for t in texts]


def make_translator(force_static: bool = False, low_threshold: int = 4,
                    cache_size: Optional[int] = 4096,
                    ngram_weight: float = 0.5) -> Translator:
    backend = get_backend(force_static=force_static)
    return Translator(Validator(backend), low_threshold=low_threshold,
                      cache_size=cache_size, ngram_weight=ngram_weight)


# ---------------------------------------------------------------------------
# Demo + CLI
# ---------------------------------------------------------------------------
DEMO_INPUTS = [
    "set a breakpoint at main",
    "break at foo.c:42",
    "set a conditional breakpoint at process_request if retries > 3",
    "put a temporary breakpoint at line 88 of parser.cpp",
    "list all breakpoints",
    "delete breakpoint 2",
    "disable breakpoint 3",
    "ignore breakpoint 1 for 5 hits",
    "run the program with args --verbose input.txt",
    "continue",
    "step over",
    "step into the function",
    "step out",
    "next instruction",
    "show a backtrace",
    "where am I",
    "select frame 2",
    "go up one frame",
    "list the threads",
    "switch to thread 3",
    "print myStruct->count + 1",
    "po self",
    "show local variables",
    "read register rax",
    "set register rax to 0x10",
    "read 32 bytes of memory at 0x7fff5fbff8c0",
    "watch the variable counter",
    "disassemble the current frame",
    "list source around line 10 of main.c",
    "attach to process named myapp",
    "attach to pid 4242",
    "kill the process",
    "set setting target.run-args to --debug",
    "show settings",
    "look up the symbol pthread_create",
    "list loaded modules",
    "help breakpoint set",
    "search the help for watchpoint",
    "quit lldb",
]


def _format_human(t: Translation, verbose: bool) -> str:
    if t.command:
        first = t.command if t.complete else "%s    # incomplete -- needs more info" % t.command
    else:
        first = str(t)
    lines = [first]
    if verbose:
        lines.append("    intent=%s confidence=%.2f valid=%s complete=%s"
                     % (t.intent, t.confidence, t.valid, t.complete))
    for n in t.notes:
        lines.append("    note: %s" % n)
    if t.suggestions and (verbose or not t.command or not t.complete):
        lines.append("    suggestions:")
        for s in t.suggestions:
            lines.append("      - %s" % s)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="nl2lldb",
        description="Translate natural-language English into valid LLDB commands.")
    p.add_argument("text", nargs="*", help="natural-language request (else read stdin)")
    p.add_argument("--json", action="store_true", help="emit JSON")
    p.add_argument("--verbose", "-v", action="store_true", help="show intent/confidence/notes")
    p.add_argument("--static", action="store_true",
                   help="force the bundled command tree (skip importing lldb)")
    p.add_argument("--demo", action="store_true", help="run a built-in demonstration set")
    p.add_argument("--version", action="version", version="nl2lldb %s" % __version__)
    args = p.parse_args(argv)

    translator = make_translator(force_static=args.static)
    backend_name = translator.validator.backend.name
    if args.verbose:
        sys.stderr.write("# validation backend: %s\n" % backend_name)

    if args.demo:
        results = translator.translate_many(DEMO_INPUTS)
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            width = max(len(s) for s in DEMO_INPUTS)
            for r in results:
                print("%-*s  ->  %s" % (width, r.input, r.command or str(r)))
        return 0

    if args.text:
        inputs = [" ".join(args.text)]
    else:
        inputs = [ln.strip() for ln in sys.stdin if ln.strip()]

    results = translator.translate_many(inputs)
    if args.json:
        payload = results[0].to_dict() if len(results) == 1 else [r.to_dict() for r in results]
        print(json.dumps(payload, indent=2))
    else:
        for r in results:
            print(_format_human(r, args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
