import os
import json
import re
import uuid
from typing import Optional
from fastapi import HTTPException, status
from groq import Groq
from sqlalchemy.orm import Session
from models import Skill, ChatSession, ChatMessage, SuggestedSkillLog
from schemas import GroqChatLLMOutput

_groq_client: Optional[Groq] = None


def _get_groq_client() -> Groq:
    """Lazily create the Groq client so a missing/blank API key doesn't
    crash the whole app at import time — it only errors when chat is
    actually used, with a clear message instead of a bare startup crash."""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GROQ_API_KEY is missing in your environment configuration."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "on", "for", "with", "as", "at", "by", "from", "this", "that",
    "it", "its", "i", "you", "your", "we", "our", "they", "he", "she", "his", "her",
    "will", "would", "can", "could", "should", "have", "has", "had", "do", "does",
    "did", "not", "no", "yes", "so", "if", "then", "than", "about", "into", "up",
    "down", "out", "over", "under", "again", "just", "also", "want", "like", "get",
    "how", "what", "why", "when", "where", "who", "which",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenizer, ignoring short/common words."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _rank_skills_by_relevance(skills: list[Skill], conversation_tokens: set[str]) -> list[Skill]:
    """Rank skills by keyword overlap with the conversation, most relevant first."""
    def score(skill: Skill) -> int:
        skill_tokens = _tokenize(f"{skill.name} {skill.description or ''}")
        return len(skill_tokens & conversation_tokens)

    return sorted(skills, key=lambda s: (-score(s), s.name))


def process_chat_session(
    session_id: str | None,
    user_message: str | None,
    db: Session,
    user_id: Optional[uuid.UUID] = None,
):
    """
    user_id is optional: anonymous users can still chat. If a valid access
    token was supplied on the request, the caller (main.py) resolves it to
    a real User and passes user.id here, which ties the session (and any
    daily-activity/streak credit recorded by the caller) to that account.
    """
    # 1. Retrieve or create session
    if not session_id:
        session_id = str(uuid.uuid4())
        session = ChatSession(id=session_id, user_id=user_id)
        db.add(session)
        db.commit()
    else:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            session = ChatSession(id=session_id, user_id=user_id)
            db.add(session)
            db.commit()
        elif user_id and not session.user_id:
            # An anonymous session later continued by a logged-in user: attach it.
            session.user_id = user_id
            db.commit()

    # 2. Fetch conversation history
    past_messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.asc())
        .all()
    )

    # 3. Fetch available skills from DB to provide to LLM context
    available_skills = db.query(Skill).all()
    skills_context = ", ".join([f"UUID '{s.id}': {s.name}" for s in available_skills])

    # 3b. Enforce a turn window: no finishing before MIN_USER_TURNS, must finish by MAX_USER_TURNS
    MIN_USER_TURNS = 6
    MAX_USER_TURNS = 10
    MIN_SUGGESTED_SKILLS = 2

    user_turn_count = sum(1 for msg in past_messages if msg.role == "user")
    if user_message and user_message.strip():
        user_turn_count += 1  # account for the message being processed right now

    below_min = user_turn_count < MIN_USER_TURNS
    force_finish = user_turn_count >= MAX_USER_TURNS

    if below_min:
        turn_instruction = (
            f"   You are only at turn {user_turn_count} of a required minimum of "
            f"{MIN_USER_TURNS} user messages. You MUST keep `is_finished` false and "
            "keep gathering context — do NOT conclude or suggest skills yet, "
            "no matter how confident you are."
        )
    elif force_finish:
        turn_instruction = (
            f"   You have reached the maximum of {MAX_USER_TURNS} user messages. "
            "You MUST conclude the conversation now: set `is_finished` to true, write a "
            "concluding `reply` based on everything discussed so far, and you MUST select "
            f"at least {MIN_SUGGESTED_SKILLS} matching skill UUIDs in `recommended_skill_ids` "
            "(pick the closest matches even if you're not fully certain)."
        )
    else:
        turn_instruction = (
            f"   You are at turn {user_turn_count} of {MAX_USER_TURNS} (minimum "
            f"{MIN_USER_TURNS} required before finishing). You may conclude now if you have "
            f"enough context, but if you do, you MUST select at least {MIN_SUGGESTED_SKILLS} "
            "matching skill UUIDs in `recommended_skill_ids`."
        )

    system_prompt = f"""
    You are an intelligent career and skill counseling chatbot.
    Your task is to converse naturally with the user, understand their goals, and assess whether you have gathered enough context to recommend skill learning paths.

    Available Skills in database:
    [{skills_context}]

    Rules:
    1. If user message is empty and history is empty, welcome them warmly and ask about their goals.
    2. If you need more context, set `is_finished` to false, keep `recommended_skill_ids` empty, and give a `reply`.
    3. When you have sufficient information or user asks for recommendations, set `is_finished` to true, provide a concluding `reply`, and select at least {MIN_SUGGESTED_SKILLS} matching skill UUID strings in `recommended_skill_ids`.
    4. The conversation must last between {MIN_USER_TURNS} and {MAX_USER_TURNS} user messages. This is turn {user_turn_count}.
{turn_instruction}

    You MUST return ONLY a valid JSON object matching this schema:
    {{
      "reply": "string message to user",
      "is_finished": true/false,
      "recommended_skill_ids": ["uuid-string-1", "uuid-string-2"]
    }}
    """

    messages = [{"role": "system", "content": system_prompt}]

    for msg in past_messages:
        messages.append({"role": msg.role, "content": msg.content})

    actual_message = user_message.strip() if user_message else "Hello! Starting new session."
    messages.append({"role": "user", "content": actual_message})

    if user_message and user_message.strip():
        db.add(ChatMessage(session_id=session_id, role="user", content=user_message.strip()))
        db.commit()

    # 4. Call Groq API
    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.3
        )
        raw_content = response.choices[0].message.content
        data = json.loads(raw_content)
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Groq returned a response that wasn't valid JSON: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Groq API call failed: {str(e)}"
        )

    llm_output = GroqChatLLMOutput(
        reply=data.get("reply", ""),
        is_finished=data.get("is_finished", False),
        recommended_skill_ids=data.get("recommended_skill_ids", [])
    )

    # 5. Hard enforcement of the turn window and minimum skill suggestions,
    #    regardless of what the LLM returned.
    if below_min:
        # Not allowed to finish yet, no matter what the LLM said.
        llm_output.is_finished = False
        llm_output.recommended_skill_ids = []
    else:
        if force_finish:
            llm_output.is_finished = True

        if llm_output.is_finished:
            # Guarantee at least MIN_SUGGESTED_SKILLS suggestions.
            chosen_ids = list(dict.fromkeys(llm_output.recommended_skill_ids))  # de-dupe, keep order
            if len(chosen_ids) < MIN_SUGGESTED_SKILLS:
                # Build the full conversation text (history + this turn + the reply)
                # so padding picks come from what was actually discussed.
                conversation_text = " ".join(
                    [msg.content for msg in past_messages]
                    + [actual_message, llm_output.reply]
                )
                conversation_tokens = _tokenize(conversation_text)

                remaining_skills = [s for s in available_skills if s.id not in chosen_ids]
                ranked_remaining = _rank_skills_by_relevance(remaining_skills, conversation_tokens)

                for skill in ranked_remaining:
                    chosen_ids.append(skill.id)
                    if len(chosen_ids) >= MIN_SUGGESTED_SKILLS:
                        break
            llm_output.recommended_skill_ids = chosen_ids

            if not llm_output.reply:
                llm_output.reply = (
                    "We've covered a lot of ground! Based on our conversation, "
                    "here are some skill paths I'd recommend to get you started."
                )

    # Save assistant response
    db.add(ChatMessage(session_id=session_id, role="assistant", content=llm_output.reply))

    # Log suggested skills to DB if finished
    if llm_output.is_finished and llm_output.recommended_skill_ids:
        for sk_id in llm_output.recommended_skill_ids:
            db.add(SuggestedSkillLog(session_id=session_id, skill_id=sk_id))

    db.commit()

    return session_id, llm_output
