"""Source registry and number formatting.

Every number on the page is rendered through `Reg.num()`, which tags it with the id of the
file it came from. The page embeds the registry as JSON; page.js shows file + date + status on
hover / focus / tap. Status drives the visual badge.
"""

from __future__ import annotations

import html
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("LLMING_ROOT") or Path(__file__).resolve().parents[2])  # repo root

# status -> (badge text, one-line meaning)
STATUS = {
    "authoritative": ("authoritative", "Full run on an idle machine (JSON flag authoritative=true)."),
    "measured": ("measured", "Real measurement; the file carries no idle-machine flag."),
    "nonidle": ("full run, not idle", "Full run, but the machine was not confirmed idle."),
    "smoke": ("smoke", "Short validation run with other work on the machine. Not a benchmark."),
    "derived": ("derived", "Computed by this report from other sources or a published formula."),
    "doc": ("from docs", "Quoted from a README or design note, not from result JSON."),
    "pending": ("in progress", "Intermediate build output from a lane that is still running."),
}


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def fmt_date(ts) -> str:
    """ISO string / epoch -> 'YYYY-MM-DD HH:MM' in local time."""
    if ts is None:
        return "unknown date"
    try:
        if isinstance(ts, (int, float)):
            d = datetime.fromtimestamp(ts)
        else:
            s = str(ts).replace("Z", "+00:00")
            d = datetime.fromisoformat(s)
            if d.tzinfo is not None:
                d = d.astimezone().replace(tzinfo=None)
        return d.strftime("%Y-%m-%d %H:%M")
    except (ValueError, OSError):
        return str(ts)


def fmt_val(v, nd=None) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "–"
    if nd is None:
        a = abs(v)
        nd = 0 if a >= 100 else (1 if a >= 10 else (2 if a >= 1 else 3))
    s = f"{v:,.{nd}f}"
    return s


@dataclass
class Source:
    id: int
    path: str  # repo-relative
    date: str
    status: str
    note: str = ""


@dataclass
class Reg:
    sources: dict = field(default_factory=dict)  # path -> Source

    def add(self, path, date=None, status="measured", note="") -> Source:
        p = Path(path)
        rel = str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(path)
        if rel in self.sources:
            return self.sources[rel]
        if date is None and p.is_absolute() and p.exists():
            date = p.stat().st_mtime
        s = Source(len(self.sources) + 1, rel, fmt_date(date), status, note)
        self.sources[rel] = s
        return s

    def num(self, v, src: Source | None, nd=None, unit="", detail="", cls="") -> str:
        """A number with its provenance. Missing values render as a dash with a reason."""
        txt = fmt_val(v, nd)
        if v is None or src is None:
            tip = detail or "No data yet"
            return f'<span class="n missing" tabindex="0" data-d="{esc(tip)}">–</span>'
        u = f'<span class="u">{esc(unit)}</span>' if unit else ""
        return (f'<span class="n st-{src.status} {cls}" tabindex="0" data-s="{src.id}"'
                f'{f" data-d={chr(34)}{esc(detail)}{chr(34)}" if detail else ""}>{txt}{u}</span>')

    def badge(self, src_or_status) -> str:
        st = src_or_status if isinstance(src_or_status, str) else src_or_status.status
        label, meaning = STATUS.get(st, (st, ""))
        return f'<span class="badge b-{esc(st)}" title="{esc(meaning)}">{esc(label)}</span>'

    def badges_for(self, srcs) -> str:
        sts = []
        for s in srcs:
            if s is not None and s.status not in sts:
                sts.append(s.status)
        return " ".join(self.badge(s) for s in sts)

    def cite(self, srcs) -> str:
        """Inline source list for figure captions."""
        out = []
        seen = set()
        for s in srcs:
            if s is None or s.id in seen:
                continue
            seen.add(s.id)
            out.append(f'<a class="cite" href="#src-{s.id}" data-s="{s.id}">[{s.id}]</a>')
        return "".join(out)

    def to_json(self) -> str:
        d = {s.id: {"p": s.path, "d": s.date, "st": STATUS.get(s.status, (s.status,))[0], "k": s.status,
                    "n": s.note} for s in self.sources.values()}
        return json.dumps(d, separators=(",", ":")).replace("</", "<\\/")
