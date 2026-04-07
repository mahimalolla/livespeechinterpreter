#!/usr/bin/env python3
"""
Send a simple CI/CD notification email via SMTP (Gmail-compatible).

Cloud Build: set secret SMTP_PASS (Gmail App Password) in Secret Manager and
grant the Cloud Build service account secretAccessor. Optional SMTP_USER
defaults to NOTIFY_TO.

Local: put SMTP_USER, SMTP_PASS, NOTIFY_TO in repo-root .env (gitignored).
  python scripts/notify_email.py --status SUCCESS --build-id test-1 --message "hello"

SMTP_USER / SMTP_PASS / NOTIFY_TO in .env always override the same keys from the shell
(so a stale `export SMTP_PASS=...` cannot mask your 16-char .env). Other keys still only
fill missing env. Cloud Build has no .env (.gcloudignore); SMTP_PASS comes from Secret Manager.
"""
from __future__ import annotations

import argparse
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[7:].strip()
    if "=" not in line:
        return None
    key, _, val = line.partition("=")
    key, val = key.strip(), val.strip()
    if not key:
        return None
    if val.startswith(("'", '"')) and len(val) >= 2 and val[-1] == val[0]:
        val = val[1:-1]
    return key, val


def _load_dotenv() -> None:
    """Load repo-root .env. SMTP_USER, SMTP_PASS, NOTIFY_TO override shell (last line wins)."""
    path = _repo_root() / ".env"
    if not path.is_file():
        return
    override_keys = frozenset({"SMTP_USER", "SMTP_PASS", "NOTIFY_TO"})
    overrides: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_dotenv_line(raw)
        if not parsed:
            continue
        key, val = parsed
        if key in override_keys:
            overrides[key] = val
        elif key not in os.environ:
            os.environ[key] = val
    for key, val in overrides.items():
        os.environ[key] = val


def main() -> int:
    _load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--to", default=os.environ.get("NOTIFY_TO", ""), help="Recipient")
    p.add_argument("--status", default="", help="e.g. SUCCESS or FAILED (required unless --print-config)")
    p.add_argument("--build-id", default=os.environ.get("BUILD_ID", "unknown"))
    p.add_argument("--commit", default=os.environ.get("SHORT_SHA", "unknown"))
    p.add_argument("--branch", default=os.environ.get("BRANCH_NAME", "unknown"))
    p.add_argument("--message", default="", help="Extra body line")
    p.add_argument(
        "--print-config",
        action="store_true",
        help="Print SMTP_USER, NOTIFY_TO, and password length (no secret), then exit",
    )
    args = p.parse_args()

    to_addr = args.to.strip()
    smtp_user = (os.environ.get("SMTP_USER") or to_addr).strip()
    # App passwords are 16 chars; users often paste with spaces — remove them
    smtp_pass = "".join(os.environ.get("SMTP_PASS", "").split())

    if args.print_config:
        print("After loading .env (SMTP_* / NOTIFY_TO from .env override shell exports):")
        print(f"  SMTP_USER     = {smtp_user!r}")
        print(f"  NOTIFY_TO/--to = {to_addr!r}")
        print(f"  SMTP_PASS len  = {len(smtp_pass)} (Gmail app passwords are exactly 16 letters, after removing spaces)")
        if len(smtp_pass) != 16:
            print(
                "\n  >>> If .env has 16 chars but you see another length, you had SMTP_PASS exported\n"
                "      in the terminal — run: unset SMTP_PASS SMTP_USER NOTIFY_TO\n"
                "      (This script now prefers .env for those three keys; retry --print-config.)\n"
                "  >>> Or fix .env: only the 16-character app password. Google → Security → App passwords"
            )
        print("\nThis SMTP_USER must be the exact address you are logged into at myaccount.google.com")
        print("when you create the App Password (same inbox, no typos, usually @gmail.com).")
        return 0

    if not to_addr:
        print("notify_email: no recipient (NOTIFY_TO / --to); skip")
        return 0
    if not smtp_pass:
        print("notify_email: SMTP_PASS not set; skip (configure Secret Manager for Cloud Build)")
        return 0
    if not args.status:
        print("notify_email: pass --status SUCCESS (or FAILED) to send mail")
        return 2

    if "gmail.com" in smtp_user.lower() or "googlemail.com" in smtp_user.lower():
        if len(smtp_pass) != 16:
            print(
                f"notify_email: SMTP_PASS has length {len(smtp_pass)}; Gmail app passwords must be "
                "exactly 16 characters (letters only, spaces in .env are OK — we remove them).\n"
                "Edit .env: delete any extra text around the password, or create a new App Password and paste only those 16 chars."
            )
            return 1

    subject = f"[livespeechinterpreter CI] Build {args.status} — {args.build_id}"
    body = (
        f"Cloud Build finished with status: {args.status}\n\n"
        f"Build ID: {args.build_id}\n"
        f"Commit: {args.commit}\n"
        f"Branch: {args.branch}\n"
    )
    if args.message:
        body += f"\n{args.message}\n"

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_addr

    ctx = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [to_addr], msg.as_string())
    except smtplib.SMTPAuthenticationError as e:
        print(
            "Gmail rejected login (535). Checklist:\n"
            "  • SMTP_USER must be the SAME Google account that created the App Password.\n"
            "  • Use an App Password (google.com → Security → 2-Step Verification → App passwords),\n"
            "    not your normal Gmail password.\n"
            "  • In .env use SMTP_PASS=xxxxxxxxxxxxxxxx (16 chars, no quotes), or paste with spaces — we strip them.\n"
            "  • Revoke old app passwords and create a new one if unsure.\n"
            "  • Workspace/school accounts may block SMTP unless admin allows it.\n"
            f"  • Debug: logging in as {smtp_user!r}, password length={len(smtp_pass)} (expect 16).\n"
            f"  • Google: https://support.google.com/mail/?p=BadCredentials\n"
            f"  • Raw error: {e!r}"
        )
        return 1

    print(f"notify_email: sent to {to_addr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
