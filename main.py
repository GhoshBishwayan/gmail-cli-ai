from __future__ import annotations

from email.utils import parseaddr

from auth import GmailAuth
from reader import GmailReader
from sender import GmailSender
from ai_reply import ReplySuggester, build_email_context
from utils import extract_plain_text_from_payload, get_header


def _parse_contact(header_value: str) -> tuple[str, str]:
    name, email = parseaddr(header_value or "")
    return (name.strip(), email.strip())


def _build_reply_inputs(msg: dict, forced_to: str | None = None) -> tuple[str, dict]:
    """
    Build the header-rich email_text plus structured identity fields
    for the reply suggester.
    """
    payload = msg.get("payload", {}) or {}
    headers = payload.get("headers", []) or []

    from_header = get_header(headers, "From") or ""
    to_header = get_header(headers, "To") or ""
    cc_header = get_header(headers, "Cc") or ""
    reply_to_header = get_header(headers, "Reply-To") or ""
    subject = get_header(headers, "Subject") or ""
    body_text = extract_plain_text_from_payload(payload) or msg.get("snippet", "") or ""

    email_text = build_email_context(
        body_text=body_text,
        from_header=from_header,
        to_header=to_header,
        subject=subject,
        cc_header=cc_header,
        reply_to_header=reply_to_header,
    )

    original_sender_name, original_sender_email = _parse_contact(from_header)
    original_receiver_name, original_receiver_email = _parse_contact(to_header)
    reply_target_name, reply_target_email = _parse_contact(reply_to_header or from_header)

    if forced_to:
        reply_target_email = forced_to.strip()
        if not reply_target_name:
            reply_target_name = original_sender_name

    reply_kwargs = {
        "replier_name": original_receiver_name,
        "replier_email": original_receiver_email,
        "original_sender_name": original_sender_name,
        "original_sender_email": original_sender_email,
        "original_receiver_name": original_receiver_name,
        "original_receiver_email": original_receiver_email,
        "reply_target_name": reply_target_name,
        "reply_target_email": reply_target_email,
        "use_agent": True,
    }
    return email_text, reply_kwargs


def _choose_reply_text(suggester: ReplySuggester, email_text: str, reply_kwargs: dict) -> str | None:
    """Return the reply text to send, or None to cancel."""
    try:
        s1, s2 = suggester.suggest_two(email_text=email_text, **reply_kwargs)
    except Exception as e:
        print(f"⚠️  Could not generate AI suggestions: {e}")
        s1, s2 = "", ""

    if s1 or s2:
        print("\n🤖 AI Reply Suggestions")
        print("=" * 60)
        if s1:
            print("(1) Formal / professional\n" + s1)
            print("-" * 60)
        if s2:
            print("(2) Warm / friendly\n" + s2)
            print("=" * 60)

    print("\nChoose reply option:")
    print("1) Send suggestion 1")
    print("2) Send suggestion 2")
    print("3) Write/edit reply manually")
    print("4) Cancel")

    pick = input("Your choice (1-4): ").strip()
    if pick == "1" and s1:
        base = s1
    elif pick == "2" and s2:
        base = s2
    elif pick == "3":
        base = ""
    else:
        print("Cancelled.")
        return None

    if base:
        print("\nSelected text (you can edit it now). Leave blank to keep as-is.")
        print("-" * 60)
        print(base)
        print("-" * 60)
        edited = input("Edit reply (or press Enter to keep): ").rstrip()
        return edited if edited else base

    manual = input("Type your reply: ").rstrip()
    if not manual:
        print("Empty reply. Cancelled.")
        return None
    return manual


def _reply_flow(
    sender: GmailSender,
    suggester: ReplySuggester,
    messages: list[dict],
    forced_to: str | None = None,
) -> None:
    """
    If forced_to is provided (Option 2), the reply will be sent ONLY to that email address.
    If forced_to is None (Option 1), reply normally using Reply-To if available, otherwise From.
    """
    if not messages:
        return

    want = input("\nReply to one of these emails now? (y/N): ").strip().lower().startswith("y")
    if not want:
        return

    try:
        idx = int(input("Which index do you want to reply to? (e.g., 1): ").strip())
    except ValueError:
        print("Invalid index.")
        return

    if idx < 1 or idx > len(messages):
        print("Invalid index.")
        return

    msg = messages[idx - 1]
    payload = msg.get("payload", {}) or {}
    headers = payload.get("headers", []) or []

    subject = get_header(headers, "Subject") or "(no subject)"
    frm = get_header(headers, "From") or "(unknown sender)"
    reply_to = get_header(headers, "Reply-To") or ""

    email_text, reply_kwargs = _build_reply_inputs(msg=msg, forced_to=forced_to)

    # Preserve the existing email-summary feature before composing a reply.
    try:
        summary = suggester.summarize(email_text[:1000])
        if summary:
            print("\n📄 Email Summary:")
            print(summary)
            print("-" * 60)
    except Exception as e:
        print(f"⚠️ Could not summarize email: {e}")

    print("\nReplying to:")
    print(f"From: {frm}")
    if reply_to:
        print(f"Reply-To: {reply_to}")
    print(f"Subject: {subject}")

    if forced_to:
        print(f"✅ IMPORTANT: This reply will be sent ONLY to: {forced_to}")
    elif reply_kwargs.get("reply_target_email"):
        print(f"Resolved reply target: {reply_kwargs['reply_target_email']}")

    reply_text = _choose_reply_text(
        suggester=suggester,
        email_text=email_text,
        reply_kwargs=reply_kwargs,
    )
    if not reply_text:
        return

    original_id = msg.get("id")
    if not original_id:
        print("❌ Missing message ID; cannot reply.")
        return

    if forced_to:
        sender.reply_to_address(
            original_message_id=original_id,
            to_address=forced_to,
            reply_text=reply_text,
        )
    else:
        sender.reply(original_message_id=original_id, reply_text=reply_text)


def main() -> None:
    auth = GmailAuth()
    reader = GmailReader(auth)
    sender = GmailSender(auth)
    suggester = ReplySuggester()

    while True:
        print("\nGmail Utility (Modules & Classes)")
        print("1) Fetch last N mails (then reply)")
        print("2) Fetch last N mails by email address (then reply ONLY to that address)")
        print("3) Send an email (with attachments)")
        print("4) Exit")
        choice = input("\nChoose an option (1-4): ").strip()

        if choice == "1":
            try:
                n = int(input("How many messages? ").strip())
            except ValueError:
                n = 5
            mark = input("Mark as read? (y/N): ").strip().lower().startswith("y")

            messages = reader.fetch_last_n(n=n, mark_as_read=mark)
            _reply_flow(sender=sender, suggester=suggester, messages=messages, forced_to=None)

        elif choice == "2":
            email_addr = input("Email address to filter (e.g., alice@example.com): ").strip()
            if not email_addr:
                print("❌ Email address cannot be empty.")
                continue

            try:
                n = int(input("How many messages? ").strip())
            except ValueError:
                n = 5
            mark = input("Mark as read? (y/N): ").strip().lower().startswith("y")

            messages = reader.fetch_last_n_by_email(email_address=email_addr, n=n, mark_as_read=mark)

            _reply_flow(sender=sender, suggester=suggester, messages=messages, forced_to=email_addr)

        elif choice == "3":
            to_addr = input("To: ").strip()
            subject = input("Subject: ").strip()
            body = input("Body: ").strip()
            attach_input = input("Attachments (comma-separated paths, blank for none): ").strip()
            attachments = [p.strip() for p in attach_input.split(",")] if attach_input else []
            sender.send(to=to_addr, subject=subject, body=body, attachments=attachments)

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
