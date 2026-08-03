import os
import re
import uuid
import json
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, status
from jose import JWTError
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from groq import Groq

import email_utils, models, security
from app.database import engine, Base, get_db
from app.models import Skill, RoadmapStep, SuggestedSkillLog, UserSavedSkill
from app.schemas import (
    # auth
    SignUpRequest, SignInRequest, AuthResponse, SignInResponse,
    RefreshRequest, TokenResponse, ForgotPasswordRequest, ForgotPasswordResponse,
    CurrentUserResponse,
    # chat / skills
    ChatRequest, ChatResponse, SuggestedSkill,
    SkillRoadmapResponse, RoadmapStepSchema,
    SaveSkillRequest, SaveSkillResponse, SavedSkillItem,
    # roadmap / challenge
    AnalyzeVideoRequest, AnalyzeVideoResponse, LessonItem,
    ChallengeRequest, CodingChallengeResponse, SkillChallengeResponse,
    # dashboard
    StepCompleteResponse, SkillProgressResponse, SkillProgressOverviewItem,
    StreakResponse, DailyTasksResponse, ActivityCalendarDay,
)
from app.services import process_chat_session
from app.dashboard_service import (
    record_daily_activity,
    get_streak_info,
    get_daily_tasks,
    get_activity_calendar,
    mark_step_complete,
    get_skill_progress,
    get_progress_overview,
    ACTIVITY_CHAT,
    ACTIVITY_CHALLENGE,
)

# -------------------------------------------------------------------
# Environment & LLM Client Setup
# -------------------------------------------------------------------
# NOTE: load_dotenv() already ran when app.database was imported above,
# so local .env values are available here too.

api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=api_key) if api_key else None

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Skill Guidance Platform API",
    description="Authentication, skill guidance chat, YouTube video roadmap analysis, coding challenges, and a student progress dashboard.",
    version="3.0.0"
)

# -------------------------------------------------------------------
# Database Startup Seeding
# -------------------------------------------------------------------
@app.on_event("startup")
def seed_database():
    db = next(get_db())
    if db.query(Skill).count() == 0:
        rn_id = str(uuid.uuid4())
        fa_id = str(uuid.uuid4())

        rn_skill = Skill(id=rn_id, name="React Native Mobile Development", description="Build cross-platform iOS/Android apps.")
        fa_skill = Skill(id=fa_id, name="FastAPI Backend Engineering", description="Build high-performance RESTful Python APIs.")

        db.add_all([rn_skill, fa_skill])
        db.commit()

        db.add_all([
            RoadmapStep(skill_id=rn_id, order=1, title="JavaScript & TypeScript Fundamentals", description="Master ES6+, async/await, and TypeScript types."),
            RoadmapStep(skill_id=rn_id, order=2, title="React Core & Hooks", description="Learn state management, useEffect, custom hooks, and component lifecycle."),
            RoadmapStep(skill_id=rn_id, order=3, title="React Native Components & Layout", description="Build layouts using Flexbox, View, Text, FlatList, and StyleSheet."),
            RoadmapStep(skill_id=rn_id, order=4, title="Offline-First Architecture", description="Implement local SQLite storage, sync logic, and state preservation."),
        ])

        db.add_all([
            RoadmapStep(skill_id=fa_id, order=1, title="Python 3.11+ & Pydantic", description="Master type hinting, dataclasses, and Pydantic schema validation."),
            RoadmapStep(skill_id=fa_id, order=2, title="Async FastAPI Core", description="Understand routing, dependency injection, and middleware."),
            RoadmapStep(skill_id=fa_id, order=3, title="Database Integration with SQLAlchemy", description="Setup SQLite ORM models and migrations."),
        ])
        db.commit()

# -------------------------------------------------------------------
# Helper Functions for YouTube & Groq
# -------------------------------------------------------------------
def extract_youtube_id(url_or_id: str) -> str:
    """Extract 11-character YouTube video ID from various URL formats."""
    regex = r"(?:v=|\/([0-9A-Za-z_-]{11}).*|list=|\/embed\/|\/v\/|youtu\.be\/|\/shorts\/)([0-9A-Za-z_-]{11})"
    match = re.search(regex, url_or_id)
    if match:
        return match.group(2) if match.group(2) else match.group(1)
    if len(url_or_id) == 11 and re.match(r"^[0-9A-Za-z_-]{11}$", url_or_id):
        return url_or_id
    raise HTTPException(status_code=400, detail="Invalid YouTube Video URL or ID.")


def fetch_youtube_transcript(video_id: str) -> str:
    """Retrieves and concatenates transcript text for a YouTube video."""
    try:
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
        except AttributeError:
            transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en', 'en-US'])

        full_text = " ".join([item['text'] for item in transcript_list])
        return full_text
    except (TranscriptsDisabled, NoTranscriptFound):
        raise HTTPException(status_code=404, detail="No public English transcript found for this video.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching transcript: {str(e)}")


def get_active_groq_client() -> Groq:
    """Ensure Groq client is properly initialized."""
    if not groq_client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GROQ_API_KEY is missing in your environment configuration (.env)."
        )
    return groq_client

# =====================================================================
# Auth Endpoints
# =====================================================================
@app.post(
    "/auth/signup",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["auth"],
)
def sign_up(payload: SignUpRequest, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.email == payload.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )

    user = models.User(
        email=payload.email,
        first_name=payload.first_name,
        last_name=payload.last_name,
        hashed_password=security.hash_password(payload.password),
        gender=payload.gender,
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )
    db.refresh(user)

    access_token = security.create_access_token(str(user.id))
    refresh_token = security.create_refresh_token(str(user.id))

    return AuthResponse(
        user_id=user.id,
        access_token=access_token,
        refresh_token=refresh_token,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        gender=user.gender,
    )


@app.post(
    "/auth/signin",
    response_model=SignInResponse,
    tags=["auth"],
)
def sign_in(payload: SignInRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == payload.email).first()

    # Use the same error for "no such user" and "wrong password" so we don't
    # leak which emails are registered.
    invalid_credentials = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password",
    )

    if not user or not security.verify_password(payload.password, user.hashed_password):
        raise invalid_credentials

    access_token = security.create_access_token(str(user.id))
    refresh_token = security.create_refresh_token(str(user.id))

    return SignInResponse(
        user_id=user.id,
        access_token=access_token,
        refresh_token=refresh_token,
        email=user.email,
    )


@app.post(
    "/auth/refresh",
    response_model=TokenResponse,
    tags=["auth"],
)
def refresh_token_endpoint(payload: RefreshRequest, db: Session = Depends(get_db)):
    try:
        claims = security.decode_token(payload.refresh_token)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    if claims.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is not a refresh token",
        )

    user_id = claims.get("sub")
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists",
        )

    return TokenResponse(
        access_token=security.create_access_token(str(user.id)),
        refresh_token=security.create_refresh_token(str(user.id)),
    )


@app.post(
    "/auth/forgot-password",
    response_model=ForgotPasswordResponse,
    tags=["auth"],
)
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == payload.email).first()

    # Generic response either way, so we don't reveal which emails are
    # registered (email enumeration protection).
    generic_message = (
        "If an account exists for this email, a temporary password has been sent."
    )

    if not user:
        return ForgotPasswordResponse(
            message=generic_message,
            url=f"{email_utils.APP_BASE_URL}/login",
        )

    temp_password = security.generate_temporary_password()
    user.hashed_password = security.hash_password(temp_password)
    db.commit()

    login_url = email_utils.send_temporary_password_email(user.email, temp_password)

    return ForgotPasswordResponse(message=generic_message, url=login_url)


@app.get("/auth/me", response_model=CurrentUserResponse, tags=["auth"])
def get_me(current_user: models.User = Depends(security.get_current_user)):
    """Returns the profile of whoever the access token belongs to."""
    return CurrentUserResponse(
        user_id=current_user.id,
        email=current_user.email,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        gender=current_user.gender,
    )


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}

# =====================================================================
# Chat
# =====================================================================
@app.post("/api/chat", response_model=ChatResponse, tags=["chat"])
def chat_endpoint(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(security.get_current_user_optional),
):
    """
    Works anonymously. If a valid access token is supplied, the chat
    session (and today's chat activity credit) is tied to that account.
    """
    session_id, llm_output = process_chat_session(
        payload.session_id,
        payload.message,
        db,
        user_id=current_user.id if current_user else None,
    )

    if current_user and payload.message and payload.message.strip():
        record_daily_activity(current_user.id, ACTIVITY_CHAT, db)

    suggested_skills: List[SuggestedSkill] = []
    if llm_output.is_finished and llm_output.recommended_skill_ids:
        skills = db.query(Skill).filter(Skill.id.in_(llm_output.recommended_skill_ids)).all()
        suggested_skills = [SuggestedSkill.model_validate(s) for s in skills]

    return ChatResponse(
        session_id=session_id,
        response=llm_output.reply,
        is_finished=llm_output.is_finished,
        suggested_skills=suggested_skills
    )

# =====================================================================
# Skills / Roadmap (public)
# =====================================================================
@app.get("/api/skills/{skill_id}/roadmap", response_model=SkillRoadmapResponse, tags=["skills"])
def get_skill_roadmap(skill_id: str, db: Session = Depends(get_db)):
    skill = db.query(Skill).filter(Skill.id == skill_id).first()
    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill with UUID '{skill_id}' not found."
        )

    steps = (
        db.query(RoadmapStep)
        .filter(RoadmapStep.skill_id == skill_id)
        .order_by(RoadmapStep.order.asc())
        .all()
    )

    return SkillRoadmapResponse(
        skill_id=skill.id,
        skill_name=skill.name,
        roadmap=[RoadmapStepSchema.model_validate(step) for step in steps]
    )


@app.get("/api/skills/suggested", response_model=List[SuggestedSkill], tags=["skills"])
def get_all_suggested_skills(session_id: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Skill).join(SuggestedSkillLog, Skill.id == SuggestedSkillLog.skill_id)

    if session_id:
        query = query.filter(SuggestedSkillLog.session_id == session_id)

    suggested_skills = query.distinct().all()
    return [SuggestedSkill.model_validate(s) for s in suggested_skills]

# =====================================================================
# Saved Skills (requires auth)
# =====================================================================
@app.post("/api/skills/save", response_model=SaveSkillResponse, tags=["skills"])
def save_favorite_skill(
    payload: SaveSkillRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user),
):
    skill = db.query(Skill).filter(Skill.id == payload.skill_id).first()
    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill with UUID '{payload.skill_id}' does not exist."
        )

    existing = (
        db.query(UserSavedSkill)
        .filter(
            UserSavedSkill.skill_id == payload.skill_id,
            UserSavedSkill.user_id == current_user.id,
        )
        .first()
    )

    if existing:
        return SaveSkillResponse(
            message="Skill is already in favorites.",
            saved_skill_id=existing.id,
            skill=SuggestedSkill.model_validate(skill)
        )

    new_saved = UserSavedSkill(skill_id=payload.skill_id, user_id=current_user.id)
    db.add(new_saved)
    db.commit()
    db.refresh(new_saved)

    return SaveSkillResponse(
        message="Skill successfully saved to favorites.",
        saved_skill_id=new_saved.id,
        skill=SuggestedSkill.model_validate(skill)
    )


@app.get("/api/skills/saved", response_model=List[SavedSkillItem], tags=["skills"])
def get_saved_skills(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user),
):
    """Retrieves all skills favorited/saved by the authenticated user."""
    saved_entries = (
        db.query(UserSavedSkill)
        .filter(UserSavedSkill.user_id == current_user.id)
        .order_by(UserSavedSkill.saved_at.desc())
        .all()
    )

    return [
        SavedSkillItem(
            saved_id=item.id,
            user_id=item.user_id,
            saved_at=item.saved_at,
            skill=SuggestedSkill.model_validate(item.skill)
        )
        for item in saved_entries
    ]

# =====================================================================
# Video Analysis & Coding Challenges
# =====================================================================
@app.post("/api/v1/roadmap/analyze-video", response_model=AnalyzeVideoResponse, tags=["roadmap"])
async def analyze_video_for_step(payload: AnalyzeVideoRequest):
    """
    Checks if a YouTube video aligns with a specific roadmap step,
    extracts core lessons, and structures the findings.
    """
    client = get_active_groq_client()
    video_id = extract_youtube_id(payload.youtube_url)
    transcript = fetch_youtube_transcript(video_id)

    truncated_transcript = transcript[:12000]

    system_prompt = (
        "You are an expert technical curriculum builder. "
        "Analyze the provided transcript against a specific roadmap learning step. "
        "Return ONLY a raw JSON object with the following keys: "
        "'is_relevant' (boolean), 'relevance_score' (int 0-100), "
        "'summary' (string), and 'key_lessons' (list of objects with 'title' and 'takeaway')."
    )

    user_prompt = f"""
    Roadmap Step: {payload.step_title}
    Step Description: {payload.step_description or 'N/A'}

    Video Transcript Excerpt:
    "{truncated_transcript}"
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )

        result = json.loads(response.choices[0].message.content)

        return AnalyzeVideoResponse(
            roadmap_id=payload.roadmap_id,
            step_title=payload.step_title,
            youtube_id=video_id,
            is_relevant=result.get("is_relevant", False),
            relevance_score=result.get("relevance_score", 0),
            key_lessons=[LessonItem(**item) for item in result.get("key_lessons", [])],
            summary=result.get("summary", "")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")


@app.post("/api/v1/roadmap/generate-challenge", response_model=CodingChallengeResponse, tags=["roadmap"])
async def generate_coding_challenge(
    payload: ChallengeRequest,
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(security.get_current_user_optional),
):
    """
    Generates a coding challenge tailored to the topic and content of the video.
    Works anonymously; contributes to the daily-challenge task/streak when logged in.
    """
    client = get_active_groq_client()
    context_text = ""

    if payload.video_summary_or_transcript:
        context_text = payload.video_summary_or_transcript[:12000]
    elif payload.youtube_url:
        video_id = extract_youtube_id(payload.youtube_url)
        context_text = fetch_youtube_transcript(video_id)[:12000]
    else:
        context_text = f"General concepts around: {payload.step_title}"

    system_prompt = (
        "You are an expert technical interviewer and coding instructor. "
        "Create a practical coding challenge based on the topic and video context provided. "
        "Return ONLY a raw JSON object with the following keys: "
        "'challenge_title' (string), 'problem_statement' (string), "
        "'starter_code' (string), 'hints' (list of strings), "
        "and 'expected_output_or_criteria' (string)."
    )

    user_prompt = f"""
    Roadmap Step: {payload.step_title}
    Difficulty: {payload.difficulty}
    Context/Content:
    "{context_text}"
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.4
        )

        result = json.loads(response.choices[0].message.content)

        if current_user:
            record_daily_activity(current_user.id, ACTIVITY_CHALLENGE, db)

        return CodingChallengeResponse(
            roadmap_id=payload.roadmap_id,
            step_title=payload.step_title,
            difficulty=payload.difficulty or "Medium",
            challenge_title=result.get("challenge_title", "Hands-on Exercise"),
            problem_statement=result.get("problem_statement", ""),
            starter_code=result.get("starter_code", ""),
            hints=result.get("hints", []),
            expected_output_or_criteria=result.get("expected_output_or_criteria", "")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")


@app.get("/api/v1/skills/{skill_id}/challenge", response_model=SkillChallengeResponse, tags=["roadmap"])
def generate_challenge_for_skill(
    skill_id: str,
    difficulty: Optional[str] = "Medium",
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(security.get_current_user_optional),
):
    """
    Generates a coding challenge for a skill using only its skill_id.
    Automatically pulls the skill's name, description, and roadmap steps
    from the database to build context for the LLM. Works anonymously;
    contributes to the daily-challenge task/streak when logged in.
    """
    client = get_active_groq_client()

    skill = db.query(Skill).filter(Skill.id == skill_id).first()
    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill with UUID '{skill_id}' not found."
        )

    steps = (
        db.query(RoadmapStep)
        .filter(RoadmapStep.skill_id == skill_id)
        .order_by(RoadmapStep.order.asc())
        .all()
    )
    steps_context = "\n".join(f"- {s.title}: {s.description}" for s in steps) or "N/A"

    system_prompt = (
        "You are an expert technical interviewer and coding instructor. "
        "Create a practical coding challenge based on the skill and its roadmap steps provided. "
        "Return ONLY a raw JSON object with the following keys: "
        "'challenge_title' (string), 'problem_statement' (string), "
        "'starter_code' (string), 'hints' (list of strings), "
        "and 'expected_output_or_criteria' (string)."
    )

    user_prompt = f"""
    Skill: {skill.name}
    Skill Description: {skill.description or 'N/A'}
    Difficulty: {difficulty}
    Roadmap Steps:
    {steps_context}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.4
        )

        result = json.loads(response.choices[0].message.content)

        if current_user:
            record_daily_activity(current_user.id, ACTIVITY_CHALLENGE, db)

        return SkillChallengeResponse(
            skill_id=skill.id,
            skill_name=skill.name,
            difficulty=difficulty or "Medium",
            challenge_title=result.get("challenge_title", "Hands-on Exercise"),
            problem_statement=result.get("problem_statement", ""),
            starter_code=result.get("starter_code", ""),
            hints=result.get("hints", []),
            expected_output_or_criteria=result.get("expected_output_or_criteria", "")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")

# =====================================================================
# Student Dashboard (all require auth — inherently user-specific)
# =====================================================================
@app.post("/api/roadmap/steps/{step_id}/complete", response_model=StepCompleteResponse, tags=["dashboard"])
def complete_roadmap_step(
    step_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user),
):
    """Marks a roadmap step as completed. Counts toward today's task checklist and the streak."""
    progress = mark_step_complete(current_user.id, step_id, db)
    if not progress:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Roadmap step '{step_id}' not found."
        )

    return StepCompleteResponse(
        step_id=step_id,
        user_id=current_user.id,
        completed=progress.completed_at is not None,
        completed_at=progress.completed_at
    )


@app.get("/api/skills/{skill_id}/progress", response_model=SkillProgressResponse, tags=["dashboard"])
def skill_progress(
    skill_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user),
):
    skill = db.query(Skill).filter(Skill.id == skill_id).first()
    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill with UUID '{skill_id}' not found."
        )

    progress = get_skill_progress(current_user.id, skill_id, db)
    return SkillProgressResponse(skill_id=skill_id, **progress)


@app.get("/api/dashboard/progress-overview", response_model=List[SkillProgressOverviewItem], tags=["dashboard"])
def progress_overview(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user),
):
    overview = get_progress_overview(current_user.id, db)
    return [SkillProgressOverviewItem(**item) for item in overview]


@app.get("/api/dashboard/streaks", response_model=StreakResponse, tags=["dashboard"])
def streak_info(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user),
):
    info = get_streak_info(current_user.id, db)
    return StreakResponse(**info)


@app.get("/api/dashboard/daily-tasks", response_model=DailyTasksResponse, tags=["dashboard"])
def daily_tasks(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user),
):
    tasks = get_daily_tasks(current_user.id, db)
    return DailyTasksResponse(**tasks)


@app.get("/api/dashboard/activity-calendar", response_model=List[ActivityCalendarDay], tags=["dashboard"])
def activity_calendar(
    days: int = 30,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_user),
):
    if days < 1 or days > 365:
        raise HTTPException(status_code=400, detail="days must be between 1 and 365.")
    calendar = get_activity_calendar(current_user.id, days, db)
    return [ActivityCalendarDay(**day) for day in calendar]
