from __future__ import annotations
import datetime as dt
from sqlalchemy import (
    Column, DateTime, Integer, String, Text, UniqueConstraint, ForeignKey, create_engine, select
)
from sqlalchemy.orm import declarative_base, Session, relationship, Mapped, mapped_column

Base = declarative_base()

class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mailbox: Mapped[str] = mapped_column(String(64), default="INBOX", index=True)
    gmail_uid: Mapped[int] = mapped_column(Integer, nullable=False)
    gmail_history_id: Mapped[int | None] = mapped_column(Integer)
    message_id: Mapped[str | None] = mapped_column(String(255), index=True)
    internal_date: Mapped[dt.datetime | None] = mapped_column(DateTime)

    from_addr: Mapped[str | None] = mapped_column(String(255))
    to_addr: Mapped[str | None] = mapped_column(String(255))
    subject: Mapped[str | None] = mapped_column(Text)
    size: Mapped[int | None] = mapped_column(Integer)

    raw_blob_uri: Mapped[str] = mapped_column(Text)
    envelope_hash: Mapped[str] = mapped_column(String(64))

    state: Mapped[str] = mapped_column(String(16), default="NEW", index=True)  # NEW|CLASSIFIED|DECODED|FAILED
    error: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)

    attachments = relationship("Attachment", back_populates="message", cascade="all, delete-orphan")
    classified = relationship("ClassifiedMessage", back_populates="message", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("mailbox", "gmail_uid", name="uq_mailbox_uid"),)

class Attachment(Base):
    __tablename__ = "attachments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"))
    part_id: Mapped[str] = mapped_column(String(64))
    filename: Mapped[str | None] = mapped_column(String(255))
    mime: Mapped[str | None] = mapped_column(String(255))
    size: Mapped[int | None] = mapped_column(Integer)
    bytes_hash: Mapped[str] = mapped_column(String(64), index=True)
    blob_uri: Mapped[str] = mapped_column(Text)
    extracted_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    message = relationship("Message", back_populates="attachments")

class ClassifiedMessage(Base):
    __tablename__ = "classified_messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"), unique=True)
    message_type: Mapped[str] = mapped_column(String(32))  # sbd_tx|status|noise
    imei: Mapped[str | None] = mapped_column(String(32))
    tx_counter: Mapped[int | None] = mapped_column(Integer)
    confidence: Mapped[int | None] = mapped_column(Integer)  # 0..100
    classified_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    message = relationship("Message", back_populates="classified")

class DecodedL0(Base):
    __tablename__ = "decoded_l0"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"), unique=True)
    station_id: Mapped[str | None] = mapped_column(String(64))
    deployment_id: Mapped[str | None] = mapped_column(String(64))
    logger_program_version: Mapped[str | None] = mapped_column(String(64))
    payload_type: Mapped[str | None] = mapped_column(String(64))
    payload_hash: Mapped[str] = mapped_column(String(64))
    l0_json: Mapped[str] = mapped_column(Text)
    decoder_version: Mapped[str] = mapped_column(String(32), default="stub-1")
    decoded_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

def get_engine(url: str):
    return create_engine(url, future=True)

def init_db(engine):
    Base.metadata.create_all(engine)

def session(url: str) -> Session:
    return Session(get_engine(url))