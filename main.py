import os
from typing import List, Optional, Literal, Any
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import db, create_document, get_documents

app = FastAPI(title="Agent Control Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------- Schemas --------------------
class Agent(BaseModel):
    name: str = Field(..., description="Agent name")
    description: Optional[str] = Field(None, description="Short description")
    system_prompt: Optional[str] = Field(None, description="System prompt / behavior")


class KnowledgeItem(BaseModel):
    agent_id: Optional[str] = Field(None, description="Associated agent id (optional)")
    kind: Literal["prompt", "faq"] = Field(..., description="Type of knowledge item")
    title: Optional[str] = Field(None, description="Title for the item")
    content: str = Field(..., description="Main content of the item")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: Optional[datetime] = None


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    message: str


# -------------------- Utility --------------------
from bson import ObjectId

def oid(obj: Any) -> str:
    return str(obj) if isinstance(obj, ObjectId) else str(obj)


def serialize(doc: dict) -> dict:
    if not doc:
        return doc
    d = {**doc}
    if "_id" in d:
        d["id"] = str(d.pop("_id"))
    # convert datetimes to iso
    for k, v in list(d.items()):
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


# -------------------- Basic Routes --------------------
@app.get("/")
def read_root():
    return {"message": "Agent Control Backend ready"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "❌ Not Set",
        "database_name": "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = os.getenv("DATABASE_NAME") or "❌ Not Set"
            response["connection_status"] = "Connected"
            try:
                response["collections"] = db.list_collection_names()
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but error: {str(e)[:60]}"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:60]}"
    return response


# -------------------- Agents --------------------
@app.post("/agents")
def create_agent(agent: Agent):
    agent_doc = agent.model_dump()
    agent_id = create_document("agent", agent_doc)
    return {"id": agent_id}


@app.get("/agents")
def list_agents():
    items = get_documents("agent")
    return [serialize(i) for i in items]


# -------------------- Knowledge --------------------
@app.post("/knowledge")
def add_knowledge(item: KnowledgeItem):
    data = item.model_dump()
    kid = create_document("knowledgeitem", data)
    return {"id": kid}


@app.get("/knowledge")
def get_knowledge(agent_id: Optional[str] = None, kind: Optional[str] = None):
    filt = {}
    if agent_id:
        filt["agent_id"] = agent_id
    if kind:
        filt["kind"] = kind
    items = get_documents("knowledgeitem", filt)
    return [serialize(i) for i in items]


# -------------------- Chat / Test Sessions --------------------
@app.post("/chat")
def chat(req: ChatRequest):
    # Create or update a test session document
    now = datetime.now(timezone.utc)
    if req.session_id:
        session = db["testsession"].find_one({"_id": ObjectId(req.session_id)})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = {
            "agent_id": req.agent_id,
            "created_at": now,
            "updated_at": now,
            "messages": []
        }
        ins = db["testsession"].insert_one(session)
        session["_id"] = ins.inserted_id

    # Append user message
    user_msg = {"role": "user", "content": req.message, "created_at": now}
    db["testsession"].update_one({"_id": session["_id"]}, {"$push": {"messages": user_msg}, "$set": {"updated_at": now}})

    # Generate a mock assistant response influenced by knowledge
    # Fetch related prompts for simple priming
    prompt_parts: List[str] = []
    if session.get("agent_id"):
        for k in db["knowledgeitem"].find({"agent_id": session["agent_id"]}):
            if k.get("kind") == "prompt":
                prompt_parts.append(k.get("content", ""))
    sys_hint = (" | ".join(prompt_parts))[:400]

    # Very simple heuristic response
    lower = req.message.strip().lower()
    if lower in ("hi", "hello", "hey"):
        reply_text = "Hello! I\'m your agent. How can I help today?"
    elif lower.endswith("?"):
        reply_text = "Here\'s what I can tell you: " + req.message
    else:
        reply_text = f"Noted. Based on context: {sys_hint[:120]}"
    reply = {"role": "assistant", "content": reply_text, "created_at": datetime.now(timezone.utc)}

    db["testsession"].update_one({"_id": session["_id"]}, {"$push": {"messages": reply}, "$set": {"updated_at": datetime.now(timezone.utc)}})

    return {
        "session_id": oid(session["_id"]),
        "reply": reply_text
    }


@app.get("/sessions")
def list_sessions(limit: int = 20):
    cur = db["testsession"].find().sort("updated_at", -1).limit(limit)
    return [serialize(s) for s in cur]


@app.get("/stats")
def stats():
    try:
        agent_count = db["agent"].count_documents({}) if db else 0
        knowledge_count = db["knowledgeitem"].count_documents({}) if db else 0
        session_count = db["testsession"].count_documents({}) if db else 0
        latest = db["testsession"].find().sort("updated_at", -1).limit(5)
        return {
            "agents": agent_count,
            "knowledge": knowledge_count,
            "sessions": session_count,
            "recent": [serialize(s) for s in latest]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
