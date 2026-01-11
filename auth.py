# database.py
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    create_engine,
    String,
    Integer,
    DateTime,
    Float,
    ForeignKey,
    Text,
    select,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DB_PATH = DATA_DIR / "app.db"
DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH.as_posix()}")

DATA_DIR.mkdir(parents=True, exist_ok=True)
(Path(os.getenv("UPLOAD_DIR", str(DATA_DIR / "uploads")))).mkdir(parents=True, exist_ok=True)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # "admin" | "worker"
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    receipts: Mapped[list["Receipt"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Receipt(Base):
    __tablename__ = "receipts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    user: Mapped["User"] = relationship(back_populates="receipts")

    image_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    image_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Datos extraídos
    emitter: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    recipient: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    currency: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)  # ARS/PYG
    date: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)  # ISO recomendado (YYYY-MM-DD)
    operation_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extracted_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "image_filename": self.image_filename,
            "image_sha256": self.image_sha256,
            "emitter": self.emitter,
            "recipient": self.recipient,
            "amount": self.amount,
            "currency": self.currency,
            "date": self.date,
            "operation_id": self.operation_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


engine = create_engine(
    DB_URL,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_user_by_username(db, username: str) -> Optional[User]:
    return db.execute(select(User).where(User.username == username)).scalar_one_or_none()


def count_receipts_by_user(db) -> list[tuple[str, int]]:
    # Devuelve (username, count)
    rows = db.execute(
        select(User.username, func.count(Receipt.id))
        .join(Receipt, Receipt.user_id == User.id, isouter=True)
        .group_by(User.username)
        .order_by(User.username.asc())
    ).all()
    return [(r[0], int(r[1])) for r in rows]


def sha256_exists(db, sha256_hex: str) -> bool:
    existing = db.execute(select(Receipt.id).where(Receipt.image_sha256 == sha256_hex)).first()
    return existing is not None
