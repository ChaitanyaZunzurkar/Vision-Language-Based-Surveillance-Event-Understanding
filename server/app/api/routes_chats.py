"""Persistent ChatGPT-style conversation API."""

from typing import Any, Dict, List, Optional
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from server.app.api.deps import get_metadata_store
from server.src.agent.grounded_agent import GroundedAgent, grounded_agent
from server.src.storage.metadata_store import MetadataStore

router = APIRouter(prefix="/chats", tags=["Chats"])


class CreateChatRequest(BaseModel):
    title: str = "New surveillance chat"
    video_id: Optional[str] = None


class MessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)


class ChatService:
    def __init__(self, store: MetadataStore, agent: GroundedAgent):
        self.store, self.agent = store, agent

    def send(self, chat_id: str, content: str) -> Dict[str, Any]:
        chat = self.store.get_conversation(chat_id)
        if not chat:
            raise HTTPException(404, "Conversation not found")
        self.store.add_message(uuid.uuid4().hex, chat_id, "user", content)
        answer = self.agent.answer(content, chat.get("video_id"), self.store.list_messages(chat_id))
        event_ids = [item["event_id"] for item in answer["evidence"]]
        assistant = self.store.add_message(uuid.uuid4().hex, chat_id, "assistant", answer["content"], event_ids)
        return {"message": assistant, "evidence": answer["evidence"]}


@router.post("")
def create_chat(request: CreateChatRequest, store: MetadataStore = Depends(get_metadata_store)):
    return store.create_conversation(uuid.uuid4().hex, request.title, request.video_id)


@router.get("")
def list_chats(search: Optional[str] = Query(None), store: MetadataStore = Depends(get_metadata_store)):
    return store.list_conversations(search)


@router.get("/{chat_id}")
def get_chat(chat_id: str, store: MetadataStore = Depends(get_metadata_store)):
    chat = store.get_conversation(chat_id)
    if not chat:
        raise HTTPException(404, "Conversation not found")
    return {**chat, "messages": store.list_messages(chat_id)}


@router.delete("/{chat_id}")
def delete_chat(chat_id: str, store: MetadataStore = Depends(get_metadata_store)):
    if not store.delete_conversation(chat_id):
        raise HTTPException(404, "Conversation not found")
    return {"deleted": True}


@router.get("/{chat_id}/messages")
def get_messages(chat_id: str, store: MetadataStore = Depends(get_metadata_store)):
    if not store.get_conversation(chat_id):
        raise HTTPException(404, "Conversation not found")
    return store.list_messages(chat_id)


@router.post("/{chat_id}/messages")
def send_message(chat_id: str, request: MessageRequest,
                 store: MetadataStore = Depends(get_metadata_store)):
    return ChatService(store, grounded_agent).send(chat_id, request.content)


search_router = APIRouter(prefix="/search", tags=["Search"])


class SearchRequest(BaseModel):
    query: str
    video_id: Optional[str] = None
    event_type: Optional[str] = None
    anomaly_category: Optional[str] = None
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    top_k: int = Field(default=10, ge=1, le=100)


@search_router.post("")
def search(request: SearchRequest, store: MetadataStore = Depends(get_metadata_store)):
    query = request.query
    result = grounded_agent.search_events(query, request.video_id, request.top_k)
    ids = {event["event_id"] for event in result["matched_events"]}
    filtered = store.get_events(request.video_id, request.event_type, request.anomaly_category,
                                request.time_start, request.time_end)
    if any(value is not None for value in (request.event_type, request.anomaly_category, request.time_start, request.time_end)):
        result["matched_events"] = [event for event in result["matched_events"] if event["event_id"] in {item.event_id for item in filtered}]
    result["total_matches"] = len(result["matched_events"])
    return result

