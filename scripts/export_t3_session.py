"""Export the T3 Code codex thread for this workspace to a readable transcript.

Reads the SQLite projection read-only and immutable, touches only
projection_thread_messages, and never opens ~/.t3/userdata/secrets.
Every message is scrubbed for credential-shaped strings before it is written.
"""
import os, re, sqlite3, sys
from pathlib import Path

DB = Path.home() / ".t3" / "userdata" / "state.sqlite"
THREAD = "24f6f99e-0e9d-430c-8556-9e7985586c2d"
OUT = Path("/home/agrov/aims-neurips/conversations/Codex/"
           "t3-code-github-connected-starter-code.md")

# Credential shapes. Each is replaced wholesale, never partially echoed.
PATTERNS = [
    (re.compile(r"-----BEGIN[^-]*PRIVATE KEY-----.*?-----END[^-]*PRIVATE KEY-----", re.S), "[REDACTED private key]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), "[REDACTED api key]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "[REDACTED github token]"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "[REDACTED hf token]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED aws key]"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"), "[REDACTED slack token]"),
    (re.compile(r"\bey[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"), "[REDACTED jwt]"),
    (re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passwd|bearer)\s*[:=]\s*[\"']?[A-Za-z0-9_\-/+]{16,}[\"']?"), r"\1=[REDACTED]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[email]"),
    (re.compile(r"/home/[A-Za-z0-9_.-]+/"), "~/"),
]


def scrub(text: str) -> tuple[str, int]:
    hits = 0
    for pattern, replacement in PATTERNS:
        text, n = pattern.subn(replacement, text)
        hits += n
    return text, hits


def main() -> int:
    if not DB.exists():
        print(f"no database at {DB}", file=sys.stderr)
        return 1
    con = sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True)
    meta = con.execute(
        "select t.title, t.created_at, t.updated_at, p.title, s.provider_name "
        "from projection_threads t "
        "join projection_projects p on p.project_id = t.project_id "
        "left join projection_thread_sessions s on s.thread_id = t.thread_id "
        "where t.thread_id = ?", (THREAD,)).fetchone()
    if meta is None:
        print("thread not found", file=sys.stderr)
        return 1
    title, created, updated, project, provider = meta
    if provider != "codex":
        print(f"refusing: provider is {provider!r}, not codex", file=sys.stderr)
        return 1

    rows = con.execute(
        "select role, text, created_at from projection_thread_messages "
        "where thread_id = ? order by created_at, message_id", (THREAD,)).fetchall()
    con.close()

    lines = [
        f"# {title}",
        "",
        f"- Source: T3 Code, provider `{provider}`",
        f"- Project: `{project}`",
        f"- Thread: `{THREAD}`",
        f"- Started: {created}",
        f"- Last updated: {updated}",
        f"- Messages: {len(rows)}",
        "",
        "> Exported from the local T3 Code SQLite projection. Credential-shaped",
        "> strings, email addresses, and absolute home paths are redacted.",
        "",
        "---",
        "",
    ]
    redactions = 0
    for role, text, when in rows:
        clean, hits = scrub(text or "")
        redactions += hits
        who = {"user": "User", "assistant": "Codex"}.get(role, role.title())
        lines += [f"## {who} — {when}", "", clean.rstrip(), "", "---", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB, {len(rows)} messages, "
          f"{redactions} redactions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
