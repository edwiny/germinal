# Capability Gaps

## 2025-01-XX: Email Sending Capability

**What I was trying to do:**
Send an email with an inspirational quote.

**Why current tooling was insufficient:**
There is no email sending tool available. The only notification tool I have is `notify_user`, which only delivers messages to the terminal/log stream. It cannot send emails via SMTP or integrate with email APIs.

**Concrete suggestion for new/improved tool:**
Create a new tool called `send_email` that:
- Takes parameters: `to` (recipient email), `subject`, `body`, and optional `from_address`
- Uses SMTP configuration from environment variables (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS)
- Includes proper email formatting (MIME/multipart support for HTML/plain text)
- Has validation for email addresses
- Returns delivery confirmation or failure details
- Risk level: "medium" (external network call)
- Consider using Python's `smtplib` or a library like `sendgrid` for delivery
