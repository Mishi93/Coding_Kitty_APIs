from datetime import date, timedelta
from sqlalchemy.orm import Session
from .models import (
    RoadmapStep,
    Skill,
    UserDailyActivity,
    UserSavedSkill,
    UserStepProgress,
    UserStreak,
)

# Activity types that count toward the daily task checklist / streak
ACTIVITY_STEP = "roadmap_step"
ACTIVITY_CHALLENGE = "challenge"
ACTIVITY_CHAT = "chat"

_ACTIVITY_COLUMN = {
    ACTIVITY_STEP: "steps_completed",
    ACTIVITY_CHALLENGE: "challenges_attempted",
    ACTIVITY_CHAT: "chat_messages_sent",
}


def _get_or_create_daily_row(user_id: str, day: date, db: Session) -> UserDailyActivity:
    row = (
        db.query(UserDailyActivity)
        .filter(UserDailyActivity.user_id == user_id, UserDailyActivity.activity_date == day)
        .first()
    )
    if not row:
        row = UserDailyActivity(user_id=user_id, activity_date=day)
        db.add(row)
        db.flush()
    return row


def _get_or_create_streak_row(user_id: str, db: Session) -> UserStreak:
    streak = db.query(UserStreak).filter(UserStreak.user_id == user_id).first()
    if not streak:
        streak = UserStreak(user_id=user_id, current_streak=0, longest_streak=0, last_active_date=None)
        db.add(streak)
        db.flush()
    return streak


def record_daily_activity(user_id: str, activity_type: str, db: Session) -> UserStreak:
    """
    Call this any time a student does something that should count toward
    today's tasks and the daily streak: completing a roadmap step,
    attempting a challenge, or sending a chat message.

    Updates (and commits) both the per-day activity tally and the
    running streak counters. Returns the updated UserStreak row.
    """
    if activity_type not in _ACTIVITY_COLUMN:
        raise ValueError(f"Unknown activity_type '{activity_type}'")

    today = date.today()

    # 1. Bump today's tally for this activity type
    daily_row = _get_or_create_daily_row(user_id, today, db)
    column = _ACTIVITY_COLUMN[activity_type]
    setattr(daily_row, column, getattr(daily_row, column) + 1)

    # 2. Update streak
    streak = _get_or_create_streak_row(user_id, db)
    if streak.last_active_date == today:
        pass  # already counted today, streak unchanged
    elif streak.last_active_date == today - timedelta(days=1):
        streak.current_streak += 1
    else:
        # First-ever activity, or a gap of >1 day: streak restarts at 1
        streak.current_streak = 1

    streak.longest_streak = max(streak.longest_streak, streak.current_streak)
    streak.last_active_date = today

    db.commit()
    db.refresh(streak)
    return streak


def get_streak_info(user_id: str, db: Session) -> dict:
    streak = db.query(UserStreak).filter(UserStreak.user_id == user_id).first()
    if not streak:
        return {
            "current_streak": 0,
            "longest_streak": 0,
            "last_active_date": None,
            "today_completed": False,
        }
    return {
        "current_streak": streak.current_streak,
        "longest_streak": streak.longest_streak,
        "last_active_date": streak.last_active_date,
        "today_completed": streak.last_active_date == date.today(),
    }


def get_daily_tasks(user_id: str, db: Session) -> dict:
    today = date.today()
    row = (
        db.query(UserDailyActivity)
        .filter(UserDailyActivity.user_id == user_id, UserDailyActivity.activity_date == today)
        .first()
    )
    steps_done = row.steps_completed if row else 0
    challenges_done = row.challenges_attempted if row else 0
    chats_done = row.chat_messages_sent if row else 0

    return {
        "date": today,
        "tasks": [
            {"type": ACTIVITY_STEP, "label": "Complete a roadmap step", "done": steps_done > 0, "count": steps_done},
            {"type": ACTIVITY_CHALLENGE, "label": "Attempt a coding challenge", "done": challenges_done > 0, "count": challenges_done},
            {"type": ACTIVITY_CHAT, "label": "Chat with the skill counselor", "done": chats_done > 0, "count": chats_done},
        ],
    }


def get_activity_calendar(user_id: str, days: int, db: Session) -> list[dict]:
    start_date = date.today() - timedelta(days=days - 1)
    rows = (
        db.query(UserDailyActivity)
        .filter(UserDailyActivity.user_id == user_id, UserDailyActivity.activity_date >= start_date)
        .all()
    )
    by_date = {r.activity_date: r for r in rows}

    calendar = []
    for i in range(days):
        day = start_date + timedelta(days=i)
        row = by_date.get(day)
        total = (row.steps_completed + row.challenges_attempted + row.chat_messages_sent) if row else 0
        calendar.append({"date": day, "activity_count": total})
    return calendar


def mark_step_complete(user_id: str, step_id: str, db: Session) -> UserStepProgress:
    step = db.query(RoadmapStep).filter(RoadmapStep.id == step_id).first()
    if not step:
        return None

    progress = (
        db.query(UserStepProgress)
        .filter(UserStepProgress.user_id == user_id, UserStepProgress.step_id == step_id)
        .first()
    )
    if not progress:
        progress = UserStepProgress(user_id=user_id, step_id=step_id)
        db.add(progress)

    already_completed = progress.completed_at is not None
    if not already_completed:
        from datetime import datetime
        progress.completed_at = datetime.utcnow()
        db.commit()
        db.refresh(progress)
        record_daily_activity(user_id, ACTIVITY_STEP, db)
    else:
        db.commit()
        db.refresh(progress)

    return progress


def get_skill_progress(user_id: str, skill_id: str, db: Session) -> dict:
    steps = (
        db.query(RoadmapStep)
        .filter(RoadmapStep.skill_id == skill_id)
        .order_by(RoadmapStep.order.asc())
        .all()
    )
    step_ids = [s.id for s in steps]

    completed_ids = set()
    if step_ids:
        completed_rows = (
            db.query(UserStepProgress)
            .filter(
                UserStepProgress.user_id == user_id,
                UserStepProgress.step_id.in_(step_ids),
                UserStepProgress.completed_at.isnot(None),
            )
            .all()
        )
        completed_ids = {r.step_id for r in completed_rows}

    total = len(steps)
    completed = len(completed_ids)
    percent = round((completed / total) * 100) if total else 0

    return {
        "total_steps": total,
        "completed_steps": completed,
        "percent_complete": percent,
        "steps": [
            {"id": s.id, "order": s.order, "title": s.title, "completed": s.id in completed_ids}
            for s in steps
        ],
    }


def get_progress_overview(user_id: str, db: Session) -> list[dict]:
    """Progress across every skill the user has saved."""
    saved = (
        db.query(UserSavedSkill)
        .filter(UserSavedSkill.user_id == user_id)
        .all()
    )
    overview = []
    for saved_skill in saved:
        skill = db.query(Skill).filter(Skill.id == saved_skill.skill_id).first()
        if not skill:
            continue
        progress = get_skill_progress(user_id, skill.id, db)
        overview.append({
            "skill_id": skill.id,
            "skill_name": skill.name,
            "total_steps": progress["total_steps"],
            "completed_steps": progress["completed_steps"],
            "percent_complete": progress["percent_complete"],
        })
    return overview
