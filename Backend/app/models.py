import uuid
import enum
from datetime import datetime

from sqlalchemy import (
    Column, String, Text, ForeignKey, Integer, DateTime, Date,
    UniqueConstraint, Enum,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.orm import relationship

from .database import Base


def generate_uuid():
    return str(uuid.uuid4())


class GUID(TypeDecorator):
    """Platform-independent UUID type.
    Uses PostgreSQL's UUID type when available, otherwise stores as CHAR(36).
    This keeps the model portable between SQLite (dev) and Postgres (prod).
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID())
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return str(value)
        if not isinstance(value, uuid.UUID):
            return str(uuid.UUID(str(value)))
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if not isinstance(value, uuid.UUID):
            return uuid.UUID(value)
        return value


class Gender(str, enum.Enum):
    male = "male"
    female = "female"
    other = "other"
    prefer_not_to_say = "prefer_not_to_say"


# ---------------------------------------------------------------------
# Auth / Users
# ---------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    gender = Column(Enum(Gender), nullable=False)

    saved_skills = relationship("UserSavedSkill", back_populates="user", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="user")
    step_progress = relationship("UserStepProgress", back_populates="user", cascade="all, delete-orphan")


# ---------------------------------------------------------------------
# Skills / Roadmaps
# ---------------------------------------------------------------------
class Skill(Base):
    __tablename__ = "skills"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)

    steps = relationship("RoadmapStep", back_populates="skill", cascade="all, delete-orphan")
    saved_by_users = relationship("UserSavedSkill", back_populates="skill", cascade="all, delete-orphan")


class RoadmapStep(Base):
    __tablename__ = "roadmap_steps"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    skill_id = Column(String(36), ForeignKey("skills.id"), nullable=False)
    order = Column(Integer, nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)

    skill = relationship("Skill", back_populates="steps")


# ---------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    # Nullable: anonymous/unauthenticated chats are still allowed. If the
    # request was made with a valid access token, this ties the session
    # (and its streak/task credit) to a real account.
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)

    session = relationship("ChatSession", back_populates="messages")


class SuggestedSkillLog(Base):
    """Tracks every skill suggested by Groq AI across chat sessions"""
    __tablename__ = "suggested_skill_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False)
    skill_id = Column(String(36), ForeignKey("skills.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    skill = relationship("Skill")


# ---------------------------------------------------------------------
# Saved skills / progress / streaks (all now tied to a real User)
# ---------------------------------------------------------------------
class UserSavedSkill(Base):
    """Tracks bookmarked/favorite skills for a user"""
    __tablename__ = "user_saved_skills"
    __table_args__ = (UniqueConstraint("user_id", "skill_id", name="uq_user_saved_skill"),)

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False)
    skill_id = Column(String(36), ForeignKey("skills.id"), nullable=False)
    saved_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="saved_skills")
    skill = relationship("Skill", back_populates="saved_by_users")


class UserStepProgress(Base):
    """Tracks whether a user has completed a given roadmap step"""
    __tablename__ = "user_step_progress"
    __table_args__ = (UniqueConstraint("user_id", "step_id", name="uq_user_step"),)

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False)
    step_id = Column(String(36), ForeignKey("roadmap_steps.id"), nullable=False)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="step_progress")
    step = relationship("RoadmapStep")


class UserDailyActivity(Base):
    """One row per user per calendar day, tallying that day's actions"""
    __tablename__ = "user_daily_activity"
    __table_args__ = (UniqueConstraint("user_id", "activity_date", name="uq_user_day"),)

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=False)
    activity_date = Column(Date, nullable=False)
    steps_completed = Column(Integer, default=0)
    challenges_attempted = Column(Integer, default=0)
    chat_messages_sent = Column(Integer, default=0)


class UserStreak(Base):
    """Running current/longest streak per user"""
    __tablename__ = "user_streaks"

    user_id = Column(GUID(), ForeignKey("users.id"), primary_key=True)
    current_streak = Column(Integer, default=0)
    longest_streak = Column(Integer, default=0)
    last_active_date = Column(Date, nullable=True)
