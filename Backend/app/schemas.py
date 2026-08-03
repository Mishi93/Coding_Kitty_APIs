import uuid
from datetime import datetime, date
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.models import Gender

# ---------------------------------------------------------------------
# Auth Schemas
# ---------------------------------------------------------------------
class SignUpRequest(BaseModel):
    email: EmailStr
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=8, max_length=128)
    gender: Gender

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c.isalpha() for c in v):
            raise ValueError("Password must contain at least one letter")
        return v


class SignInRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    user_id: uuid.UUID
    access_token: str
    refresh_token: str
    email: EmailStr
    first_name: str
    last_name: str
    gender: Gender

    class Config:
        from_attributes = True


class SignInResponse(BaseModel):
    user_id: uuid.UUID
    access_token: str
    refresh_token: str
    email: EmailStr

    class Config:
        from_attributes = True


class RefreshRequest(BaseModel):
    refresh_token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ForgotPasswordResponse(BaseModel):
    message: str
    url: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class ErrorResponse(BaseModel):
    detail: str


class CurrentUserResponse(BaseModel):
    user_id: uuid.UUID
    email: EmailStr
    first_name: str
    last_name: str
    gender: Gender

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------
# Chat Schemas
# ---------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: Optional[str] = ""


class SuggestedSkill(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""

    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    session_id: str
    response: str
    is_finished: bool
    suggested_skills: Optional[List[SuggestedSkill]] = []


class GroqChatLLMOutput(BaseModel):
    reply: str
    is_finished: bool
    recommended_skill_ids: Optional[List[str]] = []


# ---------------------------------------------------------------------
# Roadmap Schemas
# ---------------------------------------------------------------------
class RoadmapStepSchema(BaseModel):
    id: str
    order: int
    title: str
    description: str

    class Config:
        from_attributes = True


class SkillRoadmapResponse(BaseModel):
    skill_id: str
    skill_name: str
    roadmap: List[RoadmapStepSchema]


# ---------------------------------------------------------------------
# Save Skill Schemas
# ---------------------------------------------------------------------
class SaveSkillRequest(BaseModel):
    skill_id: str


class SaveSkillResponse(BaseModel):
    message: str
    saved_skill_id: str
    skill: SuggestedSkill


class SavedSkillItem(BaseModel):
    saved_id: str
    user_id: uuid.UUID
    saved_at: datetime
    skill: SuggestedSkill

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------
# Progress / Streak / Dashboard Schemas
# ---------------------------------------------------------------------
class StepCompleteResponse(BaseModel):
    step_id: str
    user_id: uuid.UUID
    completed: bool
    completed_at: Optional[datetime] = None


class RoadmapStepProgressItem(BaseModel):
    id: str
    order: int
    title: str
    completed: bool


class SkillProgressResponse(BaseModel):
    skill_id: str
    total_steps: int
    completed_steps: int
    percent_complete: int
    steps: List[RoadmapStepProgressItem]


class SkillProgressOverviewItem(BaseModel):
    skill_id: str
    skill_name: str
    total_steps: int
    completed_steps: int
    percent_complete: int


class StreakResponse(BaseModel):
    current_streak: int
    longest_streak: int
    last_active_date: Optional[date] = None
    today_completed: bool


class DailyTaskItem(BaseModel):
    type: str
    label: str
    done: bool
    count: int


class DailyTasksResponse(BaseModel):
    date: date
    tasks: List[DailyTaskItem]


class ActivityCalendarDay(BaseModel):
    date: date
    activity_count: int


# ---------------------------------------------------------------------
# Coding Challenge Schemas
# ---------------------------------------------------------------------
# --- Video Analysis Schemas ---
class AnalyzeVideoRequest(BaseModel):
    step_id: str = Field(..., description="UUID of the RoadmapStep in DB", example="step_uuid_here")
    youtube_url: str = Field(..., example="https://www.youtube.com/watch?v=dQw4w9WgXcQ")

class LessonItem(BaseModel):
    title: str
    takeaway: str

class AnalyzeVideoResponse(BaseModel):
    step_id: str
    step_title: str
    skill_name: str
    youtube_id: str
    is_relevant: bool
    relevance_score: int = Field(..., description="Relevance score from 0 to 100")
    key_lessons: List[LessonItem]
    summary: str


class ChallengeRequest(BaseModel):
    roadmap_id: str = Field(..., example="roadmap_react_native_101")
    step_title: str = Field(..., example="State Management with Redux Toolkit")
    difficulty: Optional[str] = Field("Medium", example="Medium")
    video_summary_or_transcript: Optional[str] = Field(None, description="Optional text context if video transcript was already processed.")
    youtube_url: Optional[str] = Field(None, description="Provide if transcript needs to be fetched on the fly.")


class CodingChallengeResponse(BaseModel):
    roadmap_id: str
    step_title: str
    challenge_title: str
    difficulty: str
    problem_statement: str
    starter_code: str
    hints: List[str]
    expected_output_or_criteria: str


class SkillChallengeResponse(BaseModel):
    skill_id: str
    skill_name: str
    difficulty: str
    challenge_title: str
    problem_statement: str
    starter_code: str
    hints: List[str]
    expected_output_or_criteria: str
