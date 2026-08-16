"""Authentication helpers — passwords, sessions, and agent API keys."""

from __future__ import annotations

import hashlib
import re
import secrets
from typing import Annotated

import bcrypt
from fastapi import Cookie, HTTPException, Request, status
from itsdangerous import BadSignature, SignatureExpired, TimestampSigner
from pydantic import BaseModel, EmailStr, field_validator

from web.db import MetadataStore, User

SESSION_COOKIE = "kg_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 14
AGENT_KEY_PREFIX = "kg_"
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class SignupForm(BaseModel):
    email: EmailStr
    password: str
    confirm_password: str

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        if len(value) < 8:
            raise ValueError("Password must be at least 8 characters")
        return value


class LoginForm(BaseModel):
    email: EmailStr
    password: str


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def generate_agent_key() -> str:
    return f"{AGENT_KEY_PREFIX}{secrets.token_hex(20)}"


def hash_agent_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def key_prefix(raw_key: str) -> str:
    return raw_key[:12]


class SessionManager:
    """Signed cookie session management."""

    def __init__(self, secret: str) -> None:
        self._signer = TimestampSigner(secret, salt="kg-web-session")

    def create_session_token(self, user_id: str) -> str:
        return self._signer.sign(user_id.encode()).decode()

    def read_session_token(self, token: str) -> str | None:
        try:
            user_id = self._signer.unsign(token, max_age=SESSION_MAX_AGE)
        except (BadSignature, SignatureExpired):
            return None
        return user_id.decode()


def get_or_create_session_secret(store: MetadataStore) -> str:
    existing = store.get_meta("session_secret")
    if existing:
        return existing
    secret = secrets.token_hex(32)
    store.set_meta("session_secret", secret)
    return secret


def validate_email_format(email: str) -> bool:
    return bool(EMAIL_PATTERN.match(email))


class AuthContext:
    """Request-scoped authentication dependencies."""

    def __init__(self, store: MetadataStore, sessions: SessionManager) -> None:
        self.store = store
        self.sessions = sessions

    def get_current_user(
        self,
        request: Request,
        session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
    ) -> User:
        if not session:
            raise HTTPException(status_code=status.HTTP_303_SEE_OTHER, headers={"Location": "/login"})
        user_id = self.sessions.read_session_token(session)
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_303_SEE_OTHER, headers={"Location": "/login"})
        user = self.store.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=status.HTTP_303_SEE_OTHER, headers={"Location": "/login"})
        request.state.user = user
        return user


def extract_agent_key(request: Request) -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token.startswith(AGENT_KEY_PREFIX):
            return token
    agent_key = request.headers.get("X-Agent-Key", "").strip()
    if agent_key.startswith(AGENT_KEY_PREFIX):
        return agent_key
    return None


__all__ = [
    "AGENT_KEY_PREFIX",
    "AuthContext",
    "LoginForm",
    "SESSION_COOKIE",
    "SESSION_MAX_AGE",
    "SessionManager",
    "SignupForm",
    "extract_agent_key",
    "generate_agent_key",
    "get_or_create_session_secret",
    "hash_agent_key",
    "hash_password",
    "key_prefix",
    "validate_email_format",
    "verify_password",
]
