# auth.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import bcrypt

from database import SessionLocal, init_db, get_user_by_username, User


ADMIN_DEFAULT_USERNAME = "admin_utara"
ADMIN_DEFAULT_PASSWORD = "utara2026"


@dataclass
class AuthUser:
    id: int
    username: str
    role: str  # "admin" | "worker"


def hash_password(plain_password: str) -> str:
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def ensure_default_admin() -> None:
    """
    Crea el admin por defecto si no existe.
    """
    init_db()
    with SessionLocal() as db:
        existing = get_user_by_username(db, ADMIN_DEFAULT_USERNAME)
        if existing:
            return
        admin = User(
            username=ADMIN_DEFAULT_USERNAME,
            password_hash=hash_password(ADMIN_DEFAULT_PASSWORD),
            role="admin",
        )
        db.add(admin)
        db.commit()


def authenticate(username: str, password: str) -> Optional[AuthUser]:
    with SessionLocal() as db:
        user = get_user_by_username(db, username)
        if not user:
            return None
        if not verify_password(password, user.password_hash):
            return None
        return AuthUser(id=user.id, username=user.username, role=user.role)


def create_worker(username: str, password: str) -> None:
    with SessionLocal() as db:
        if get_user_by_username(db, username):
            raise ValueError("El usuario ya existe.")
        u = User(username=username, password_hash=hash_password(password), role="worker")
        db.add(u)
        db.commit()


def delete_user(username: str) -> None:
    if username == ADMIN_DEFAULT_USERNAME:
        raise ValueError("No se puede eliminar el administrador por defecto.")
    with SessionLocal() as db:
        user = get_user_by_username(db, username)
        if not user:
            raise ValueError("Usuario no encontrado.")
        db.delete(user)
        db.commit()
