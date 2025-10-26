import os
import hashlib
import hmac
import binascii
from typing import Tuple

from .models import User
from .database_manager import load_user, save_user


def _hash_password(password: str, iterations: int = 100_000) -> str:
    """Return a PBKDF2-SHA256 hash string: pbkdf2_sha256$iter$salt$hash"""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${binascii.hexlify(salt).decode()}${binascii.hexlify(dk).decode()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algo, iter_s, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(iter_s)
        salt = binascii.unhexlify(salt_hex)
        expected = binascii.unhexlify(hash_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


def signup_user(username: str, password: str) -> Tuple[bool, str]:
    username = (username or "").strip()
    if not username or not password:
        return False, "Username and password are required."

    existing = load_user(username)
    if existing:
        return False, "User already exists."

    hashed = _hash_password(password)
    user = User(username=username, password_hash=hashed)
    save_user(user)
    return True, "Account created. Please log in."


def login_user(username: str, password: str) -> Tuple[bool, str]:
    data = load_user((username or "").strip())
    if not data:
        return False, "Invalid username or password."
    stored_hash = data.get("passwordHash", "")
    if _verify_password(password or "", stored_hash):
        return True, "Login successful."
    return False, "Invalid username or password."
