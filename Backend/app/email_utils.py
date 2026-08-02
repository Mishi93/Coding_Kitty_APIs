import logging
import os
import smtplib
from email.message import EmailMessage

logger = logging.getLogger("auth.email")

# --- Configuration -------------------------------------------------
# Set these as environment variables in production. If SMTP_HOST is not
# set, emails are logged to the console instead of actually being sent —
# handy for local development without needing a real mail server.
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", "no-reply@example.com")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

# Base URL of your frontend/login page, used to build the reset link.
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:3000")


def send_temporary_password_email(to_email: str, temporary_password: str) -> str:
    """Send (or log) an email containing a temporary password and a login URL.
    Returns the URL that was included, so the caller can also return it if needed.
    """
    login_url = f"{APP_BASE_URL}/login"

    subject = "Your temporary password"
    body = (
        f"Hi,\n\n"
        f"A password reset was requested for this account.\n\n"
        f"Temporary password: {temporary_password}\n\n"
        f"Sign in here: {login_url}\n\n"
        f"For your security, please sign in and change this password as soon as possible.\n"
        f"If you did not request this, please contact support immediately.\n"
    )

    if not SMTP_HOST:
        # Dev fallback: no SMTP configured, just log it so you can see the
        # temp password/link during local testing.
        logger.warning(
            "SMTP not configured — logging email instead of sending.\n"
            "To: %s\nSubject: %s\n%s",
            to_email, subject, body,
        )
        return login_url

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM_EMAIL
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        if SMTP_USE_TLS:
            server.starttls()
        if SMTP_USERNAME and SMTP_PASSWORD:
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)

    return login_url
