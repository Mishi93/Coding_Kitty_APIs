import os
import secrets
import string
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.database import get_db

# --- Configuration -----------------------------------------------------
# In production, set these via environment variables / a secrets manager.
# Never commit real secret keys to source control.
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_dev_only_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bearer-token schemes used by the auth dependencies below.
_bearer_scheme = HTTPBearer()
_bearer_scheme_optional = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def _create_token(subject: str, token_type: Literal["access", "refresh"], expires_delta: timedelta) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_access_token(user_id: str) -> str:
    return _create_token(user_id, "access", timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))


def create_refresh_token(user_id: str) -> str:
    return _create_token(user_id, "refresh", timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))


def decode_token(token: str) -> dict:
    """Decode and validate a JWT. Raises JWTError on failure/expiry."""
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


def generate_temporary_password(length: int = 12) -> str:
    """Generate a random temporary password containing letters, digits,
    and punctuation, guaranteed to satisfy the signup password policy
    (at least one letter and one digit)."""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    while True:
        pwd = "".join(secrets.choice(alphabet) for _ in range(length))
        if any(c.isdigit() for c in pwd) and any(c.isalpha() for c in pwd):
            return pwd


# ---------------------------------------------------------------------
# FastAPI dependencies for protected routes
# ---------------------------------------------------------------------
def _resolve_user_from_token(token: str, db: Session):
    """Shared logic: decode an access token and load the matching User."""
    from app import models  # local import avoids a circular import with models.py

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        claims = decode_token(token)
    except JWTError:
        raise credentials_exception

    if claims.get("type") != "access":
        raise credentials_exception

    sub = claims.get("sub")
    if not sub:
        raise credentials_exception

    try:
        user_uuid = uuid.UUID(sub)
    except (ValueError, TypeError):
        raise credentials_exception

    user = db.query(models.User).filter(models.User.id == user_uuid).first()
    if not user:
        raise credentials_exception

    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
):
    """Require a valid access token; 401s if missing/invalid/expired.
    Use this on routes that must be tied to a real logged-in user
    (saving skills, marking progress, streaks, dashboard, etc.)."""
    return _resolve_user_from_token(credentials.credentials, db)


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme_optional),
    db: Session = Depends(get_db),
):
    """Like get_current_user, but returns None instead of raising when no
    token (or an invalid one) is provided. Use this on routes that should
    still work anonymously but get extra behavior when logged in (chat)."""
    if credentials is None:
        return None
    try:
        return _resolve_user_from_token(credentials.credentials, db)
    except HTTPException:
        return None
