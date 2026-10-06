import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, ArrowUp, ArrowUpRight, Camera,
  Check, ChevronDown, Clapperboard, Clock3, FileJson2,
  Film, FolderOpen, LoaderCircle, Menu, MessageSquare, Paperclip,
  Plus, Search, Shield, Trash2, Upload, Video, X,
} from "lucide-react";

const API = "";
const STORAGE_KEY = "vista-chat-conversations-v1";

async function request(url, options) {
  const response = await fetch(`${API}${url}`, options);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch { /* response may not be JSON */ }
    throw new Error(message);
  }
  return response.json();
}

function newConversation() {
  return { id: crypto.randomUUID(), title: "New surveillance chat", messages: [], updatedAt: Date.now() };
}

function formatTime(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  return `${Math.floor(total / 60).toString().padStart(2, "0")}:${(total % 60).toString().padStart(2, "0")}`;
}

function readConversations() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    return Array.isArray(saved) ? saved : [];
  } catch { return []; }
}

export default function App() {
  const [initialSession] = useState(() => {
    const fresh = newConversation();
    const previous = readConversations().filter((item) => item.messages?.length);
    return { conversations: [fresh, ...previous], conversationId: fresh.id };
  });
  const [conversations, setConversations] = useState(initialSession.conversations);
  const [conversationId, setConversationId] = useState(initialSession.conversationId);
  const [videos, setVideos] = useState([]);
  const [events, setEvents] = useState([]);
  const [selectedEventId, setSelectedEventId] = useState(null);
  const [eventsLoading, setEventsLoading] = useState(false);
  const [eventsError, setEventsError] = useState("");
  const [activeView, setActiveView] = useState("chat");
  const [scope, setScope] = useState("");
  const [query, setQuery] = useState("");
  const [searchChats, setSearchChats] = useState("");
  const [working, setWorking] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeClip, setActiveClip] = useState(null);
  const [showArchive, setShowArchive] = useState(false);
  const [eventsReviewOpen, setEventsReviewOpen] = useState(false);
  const [eventsFile, setEventsFile] = useState(null);
  const [eventsPreview, setEventsPreview] = useState(null);
  const [eventsTargetVideoId, setEventsTargetVideoId] = useState("");
  const [eventsImporting, setEventsImporting] = useState(false);
  const [eventsReviewError, setEventsReviewError] = useState("");
  const [notice, setNotice] = useState("");
  const videoInput = useRef(null);
  const bottomRef = useRef(null);

  const current = conversations.find((item) => item.id === conversationId) || null;
  const messages = current?.messages || [];

  useEffect(() => {
    const saved = conversations.filter((item) => item.messages.length > 0);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(saved));
  }, [conversations]);

  useEffect(() => {
    Promise.allSettled([request("/api/videos")]).then(([videoResult]) => {
      if (videoResult.status === "fulfilled") setVideos(videoResult.value);
    });
  }, []);

  useEffect(() => {
    if (!videos.some((video) => video.status === "processing")) return undefined;
    const timer = setInterval(async () => {
      try {
        const currentVideos = await request("/api/videos");
        setVideos(currentVideos);
      } catch { /* keep the last visible state while the server recovers */ }
    }, 2200);
    return () => clearInterval(timer);
  }, [videos]);

  useEffect(() => {
    if (activeView !== "events") return undefined;
    let cancelled = false;
    setEventsLoading(true);
    setEventsError("");
    const params = scope ? `?video_id=${encodeURIComponent(scope)}` : "";
    request(`/api/events${params}`).then((items) => {
      if (cancelled) return;
      setEvents(items);
      setSelectedEventId((previous) => items.some((item) => item.event_id === previous) ? previous : items[0]?.event_id || null);
    }).catch((err) => {
      if (!cancelled) setEventsError(err.message);
    }).finally(() => {
      if (!cancelled) setEventsLoading(false);
    });
    return () => { cancelled = true; };
  }, [activeView, scope]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, working]);

  const visibleConversations = useMemo(() => {
    const search = searchChats.trim().toLowerCase();
    return [...conversations]
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .filter((item) => !search || item.title.toLowerCase().includes(search));
  }, [conversations, searchChats]);

  function updateConversation(id, updater) {
    setConversations((items) => items.map((item) => item.id === id
      ? { ...updater(item), updatedAt: Date.now() }
      : item));
  }

  function addMessage(message, id = conversationId) {
    if (!id) return;
    updateConversation(id, (item) => ({ ...item, messages: [...item.messages, { id: crypto.randomUUID(), ...message }] }));
  }

  function startConversation() {
    const next = newConversation();
    setConversations((items) => [next, ...items]);
    setConversationId(next.id);
    setQuery("");
    setError("");
    setActiveView("chat");
    setSidebarOpen(false);
  }

  function selectConversation(id) {
    setConversationId(id);
    setActiveView("chat");
    setError("");
    setSidebarOpen(false);
  }

  function deleteConversation(id) {
    const remaining = conversations.filter((item) => item.id !== id);
    if (id === conversationId) {
      const fresh = newConversation();
      setConversations([fresh, ...remaining]);
      setConversationId(fresh.id);
    } else {
      setConversations(remaining);
    }
  }

  async function askQuestion(value = query) {
    const text = value.trim();
    if (!text || working) return;
    let chatId = conversationId;
    if (!chatId) {
      const next = newConversation();
      chatId = next.id;
      setConversationId(chatId);
      setConversations((items) => [next, ...items]);
    }
    updateConversation(chatId, (item) => ({
      ...item,
      title: item.messages.length === 0 ? text.slice(0, 42) : item.title,
      messages: [...item.messages, { id: crypto.randomUUID(), role: "user", text }],
    }));
    setQuery("");
    setError("");
    setWorking(true);
    try {
      const result = await request("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text, top_k: 6, min_score: 0.08, video_id: scope || null }),
      });
      updateConversation(chatId, (item) => ({
        ...item,
        title: item.messages.length === 1 ? text.slice(0, 42) : item.title,
        messages: [...item.messages, { id: crypto.randomUUID(), role: "assistant", text: result.grounded_summary, result }],
      }));
    } catch (err) {
      setError(err.message);
      updateConversation(chatId, (item) => ({
        ...item,
        messages: [...item.messages, { id: crypto.randomUUID(), role: "assistant", text: `I couldn't search the archive. ${err.message}` }],
      }));
    } finally {
      setWorking(false);
    }
  }

  async function uploadVideo(file) {
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      const body = new FormData();
      body.append("file", file);
      body.append("auto_run", "false");
      const video = await request("/api/videos/upload", { method: "POST", body });
      setVideos((items) => [video, ...items.filter((item) => item.video_id !== video.video_id)]);
      setScope(video.video_id);
      let chatId = conversationId;
      if (!chatId) {
        const next = newConversation();
        chatId = next.id;
        setConversationId(chatId);
        setConversations((items) => [next, ...items]);
      }
      updateConversation(chatId, (item) => ({
        ...item,
        messages: [...item.messages, { id: crypto.randomUUID(), role: "assistant", videoId: video.video_id, text: `Added “${video.filename}” to your footage library.` }],
      }));
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
      if (videoInput.current) videoInput.current.value = "";
    }
  }

  async function startProcessing(videoId) {
    setError("");
    try {
      await request(`/api/pipeline/run/${videoId}?run_in_background=true`, { method: "POST" });
      setVideos((items) => items.map((video) => video.video_id === videoId ? { ...video, status: "processing" } : video));
    } catch (err) {
      if (activeView === "events") setEventsError(err.message);
      else setError(err.message);
    }
  }

  async function deleteVideo(videoId) {
    const video = videos.find((item) => item.video_id === videoId);
    if (!video || video.status === "processing") return;
    if (!window.confirm(`Delete “${video.filename}” and its events and evidence clips?`)) return;
    try {
      await request(`/api/videos/${videoId}`, { method: "DELETE" });
      setVideos((items) => items.filter((item) => item.video_id !== videoId));
      setEvents((items) => items.filter((item) => item.video_id !== videoId));
      setSelectedEventId((currentEventId) => events.find((item) => item.event_id === currentEventId)?.video_id === videoId ? null : currentEventId);
      if (scope === videoId) setScope("");
      setNotice(`Deleted ${video.filename}.`);
    } catch (err) {
      if (activeView === "events") setEventsError(err.message);
      else setError(err.message);
    }
  }

  function openEventsReview(videoId = "") {
    setEventsTargetVideoId(videoId || scope);
    setEventsFile(null);
    setEventsPreview(null);
    setEventsReviewError("");
    setEventsReviewOpen(true);
  }

  async function inspectEventsFile(file) {
    setEventsFile(file || null);
    setEventsPreview(null);
    setEventsReviewError("");
    if (!file) return;
    try {
      const payload = JSON.parse(await file.text());
      let format = "events";
      let entries = [];
      if (Array.isArray(payload?.windows)) {
        format = "windows";
        entries = payload.windows.flatMap((window, windowIndex) =>
          Array.isArray(window?.events) ? window.events.map((event) => ({ event, windowIndex })) : [],
        );
      } else if (Array.isArray(payload?.events)) {
        entries = payload.events.map((event, eventIndex) => ({ event, eventIndex }));
      } else {
        throw new Error("Expected an events array or a windows array containing events.");
      }
      const invalidCount = entries.filter(({ event }) => {
        if (!event || typeof event !== "object") return true;
        const start = Number(event.start_sec ?? event.start);
        const end = Number(event.end_sec ?? event.end);
        return !(event.event_type || event.type) || !event.description ||
          !Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end < start;
      }).length;
      const sourceVideoIds = format === "windows"
        ? [...new Set(payload.windows.map((window) => window?.video_id).filter((id) => id != null).map(String))]
        : [...new Set(payload.events.map((event) => event?.video_id).filter((id) => id != null).map(String))];
      setEventsPreview({
        format,
        count: entries.length,
        invalidCount,
        sourceVideoIds,
        sample: entries.slice(0, 3).map(({ event }) => event),
      });
    } catch (err) {
      setEventsReviewError(err instanceof SyntaxError ? "This file is not valid JSON." : err.message);
    }
  }

  async function importCheckedEvents() {
    if (!eventsFile || !eventsTargetVideoId || !eventsPreview || eventsPreview.invalidCount > 0) return;
    const body = new FormData();
    body.append("video_id", eventsTargetVideoId);
    body.append("file", eventsFile);
    setEventsImporting(true);
    setEventsReviewError("");
    try {
      const result = await request("/api/events/import", { method: "POST", body });
      const message = `Imported ${result.imported_events} event${result.imported_events === 1 ? "" : "s"} for ${videos.find((video) => video.video_id === eventsTargetVideoId)?.filename || "the selected video"}.`;
      setNotice(message);
      addMessage({ role: "assistant", text: message });
      setEventsReviewOpen(false);
      setEventsFile(null);
      setEventsPreview(null);
      setVideos(await request("/api/videos"));
      if (activeView === "events") {
        const params = scope ? `?video_id=${encodeURIComponent(scope)}` : "";
        setEvents(await request(`/api/events${params}`));
      }
    } catch (err) { setEventsReviewError(err.message); }
    finally { setEventsImporting(false); }
  }

  const eventsTarget = videos.find((video) => video.video_id === eventsTargetVideoId);
  const targetMismatch = Boolean(eventsTarget && eventsPreview?.sourceVideoIds?.length &&
    !eventsPreview.sourceVideoIds.every((id) => [eventsTarget.video_id, eventsTarget.filename.replace(/\.[^.]+$/, "")].includes(id)));

  const selectedVideo = videos.find((video) => video.video_id === scope) || null;
  const selectedEvent = events.find((event) => event.event_id === selectedEventId) || null;
  const playerVideo = videos.find((video) => video.video_id === selectedEvent?.video_id) || null;

  return (
    <div className="app-shell">
      <button className={`sidebar-scrim ${sidebarOpen ? "visible" : ""}`} onClick={() => setSidebarOpen(false)} aria-label="Close navigation" />
      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="sidebar-brand">
          <div className="brand-mark"><Camera size={18} strokeWidth={2.1} /></div>
          <div><div className="brand-name">VISTA</div></div>
          <button className="icon-button sidebar-close" onClick={() => setSidebarOpen(false)} aria-label="Close menu"><X size={18} /></button>
        </div>

        <button className="new-chat-button" onClick={startConversation}><Plus size={17} /><span>New conversation</span></button>

        <label className="sidebar-search"><Search size={15} /><input value={searchChats} onChange={(event) => setSearchChats(event.target.value)} placeholder="Search conversations" /></label>

        <div className="sidebar-section-heading"><span>FOOTAGE</span><button className="icon-button small" onClick={() => setShowArchive((value) => !value)} aria-label="Toggle archive"><FolderOpen size={15} /></button></div>
        {showArchive && <div className="archive-list">
          {videos.length === 0 && <div className="sidebar-empty">No footage yet.</div>}
          {videos.map((video) => <div className="archive-video-row" key={video.video_id}>
            <button className={`archive-video ${scope === video.video_id ? "selected" : ""}`} onClick={() => { setScope(video.video_id); setSidebarOpen(false); }}>
              <Film size={15} /><span className="archive-video-name">{video.filename}</span><span className={`status-dot ${video.status}`} />
            </button>
            <button className="row-delete-button" title={`Delete ${video.filename}`} aria-label={`Delete ${video.filename}`} disabled={video.status === "processing"} onClick={() => deleteVideo(video.video_id)}><Trash2 size={14} /></button>
          </div>)}
        </div>}

        <div className="sidebar-section-heading conversations-heading"><span>RECENT CONVERSATIONS</span><span className="conversation-count">{visibleConversations.length}</span></div>
        <div className="conversation-list">
          {visibleConversations.map((item) => <div className="conversation-row" key={item.id}>
            <button className={`conversation-item ${item.id === conversationId && activeView === "chat" ? "active" : ""}`} onClick={() => selectConversation(item.id)}><MessageSquare size={15} /><span>{item.title}</span></button>
            <button className="row-delete-button" title="Delete conversation" aria-label={`Delete conversation ${item.title}`} onClick={() => deleteConversation(item.id)}><Trash2 size={14} /></button>
          </div>)}
          {visibleConversations.length === 0 && <div className="sidebar-empty">No conversations</div>}
        </div>

        <div className="sidebar-bottom">
          <div className="scope-label"><span>SEARCH SCOPE</span><ChevronDown size={14} /></div>
          <select className="scope-select" value={scope} onChange={(event) => setScope(event.target.value)} aria-label="Search footage scope">
            <option value="">Entire video archive</option>
            {videos.map((video) => <option key={video.video_id} value={video.video_id}>{video.filename}</option>)}
          </select>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-button menu-button" onClick={() => setSidebarOpen(true)} aria-label="Open menu"><Menu size={19} /></button>
            <div className="page-title">{activeView === "events" ? "Events" : "Event search"}</div>
          </div>
          <div className="topbar-right">
            <button className={`view-switch ${activeView === "events" ? "active" : ""}`} onClick={() => { setActiveView(activeView === "events" ? "chat" : "events"); setSidebarOpen(false); }}><Film size={15} /><span>{activeView === "events" ? "Chat" : "Events"}</span></button>
            <button className="check-json-button" onClick={() => openEventsReview()}><FileJson2 size={15} /><span>Check events.json</span></button>
            <button className="upload-top-button" onClick={() => videoInput.current?.click()} disabled={uploading}><Upload size={15} /><span>{uploading ? "Adding footage…" : "Add footage"}</span></button>
          </div>
        </header>

        {activeView === "events" ? <EventsLibrary
          events={events}
          videos={videos}
          selectedEvent={selectedEvent}
          playerVideo={playerVideo}
          loading={eventsLoading}
          error={eventsError}
          onSelect={setSelectedEventId}
          onCheckJson={() => openEventsReview()}
          onDeleteVideo={deleteVideo}
        /> : <section className="conversation-stage">
          <div className="thread-scroll">
            {messages.length === 0 ? <Welcome onPrompt={(prompt) => askQuestion(prompt)} onUpload={() => videoInput.current?.click()} /> : <div className="message-thread">
      {messages.map((message) => <Message key={message.id} message={message} videos={videos} onProcess={startProcessing} onImport={() => openEventsReview(message.videoId)} activeClip={activeClip} onPlay={setActiveClip} />)}
              {working && <div className="assistant-row"><div className="assistant-avatar"><Activity size={15} /></div><div className="thinking"><span /><span /><span /></div><span className="thinking-label">Searching verified events…</span></div>}
            </div>}
            <div ref={bottomRef} />
          </div>

      <div className="composer-wrap">
            {selectedVideo && <div className="scope-chip"><Video size={14} /><span>Searching in <strong>{selectedVideo.filename}</strong></span><button onClick={() => setScope("")} aria-label="Clear footage filter"><X size={13} /></button></div>}
            {notice && <div className="notice-banner"><span>{notice}</span><button onClick={() => setNotice("")} aria-label="Dismiss"><X size={14} /></button></div>}
            {error && <div className="error-banner"><span>{error}</span><button onClick={() => setError("")} aria-label="Dismiss"><X size={14} /></button></div>}
            <form className="composer" onSubmit={(event) => { event.preventDefault(); askQuestion(); }}>
              <button type="button" className="composer-attach" onClick={() => videoInput.current?.click()} title="Add surveillance footage" disabled={uploading}><Paperclip size={19} /></button>
              <textarea value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Ask about activity, people, or what happened…" rows={1} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); askQuestion(); } }} />
              <div className="composer-actions"><button className="send-button" type="submit" disabled={!query.trim() || working} aria-label="Send message">{working ? <LoaderCircle size={17} className="spin" /> : <ArrowUp size={18} />}</button></div>
            </form>
          </div>
        </section>}
      </main>

      <input ref={videoInput} className="hidden-input" type="file" accept="video/mp4,video/avi,video/quicktime,video/x-matroska,.mp4,.avi,.mov,.mkv" onChange={(event) => uploadVideo(event.target.files?.[0])} />
      {eventsReviewOpen && <EventsReviewModal
        videos={videos}
        file={eventsFile}
        preview={eventsPreview}
        selectedVideoId={eventsTargetVideoId}
        error={eventsReviewError}
        importing={eventsImporting}
        mismatch={targetMismatch}
        onFile={inspectEventsFile}
        onSelectVideo={setEventsTargetVideoId}
        onClose={() => setEventsReviewOpen(false)}
        onImport={importCheckedEvents}
      />}
    </div>
  );
}

function Welcome({ onPrompt, onUpload }) {
  const prompts = [
    { title: "Suspicious activity", prompt: "Show suspicious or unusual activity" },
    { title: "Event timeline", prompt: "What happened in the footage?" },
    { title: "Incidents involving a person", prompt: "Find incidents involving a person" },
  ];
  return <div className="welcome-wrap">
    <div className="welcome-content">
      <div className="welcome-emblem"><Activity size={23} strokeWidth={1.8} /><span /></div>
      <h1>Search recorded<br /><span>surveillance events.</span></h1>
      <div className="welcome-actions">
        <button className="welcome-upload" onClick={onUpload}><Upload size={16} />Add surveillance footage</button>
      </div>
      <div className="prompt-grid">
        {prompts.map((item) => <button className="prompt-card" key={item.title} onClick={() => onPrompt(item.prompt)}>{item.title}</button>)}
      </div>
    </div>
  </div>;
}

function EventsLibrary({ events, videos, selectedEvent, playerVideo, loading, error, onSelect, onCheckJson, onDeleteVideo }) {
  const videoUrl = selectedEvent && playerVideo
    ? `${playerVideo.stream_url}#t=${Math.max(0, selectedEvent.start_sec)},${Math.max(selectedEvent.start_sec, selectedEvent.end_sec)}`
    : "";
  return <section className="events-library-stage">
    <div className="events-view-toolbar">
      <div><h1>Events</h1><span>{events.length} recorded</span></div>
      {playerVideo && <button className="delete-video-button" onClick={() => onDeleteVideo(playerVideo.video_id)} disabled={playerVideo.status === "processing"}><Trash2 size={14} />Delete video</button>}
    </div>
    <div className="events-library-grid">
      <div className="events-player-pane">
        {videoUrl ? <video key={`${selectedEvent.event_id}-${videoUrl}`} controls preload="metadata" src={videoUrl} /> : <div className="events-player-empty"><Film size={22} /><span>Select an event to view its video.</span></div>}
        {selectedEvent && <div className="player-event-meta">
          <div><strong>{videos.find((video) => video.video_id === selectedEvent.video_id)?.filename || selectedEvent.video_id}</strong><span>{formatTime(selectedEvent.start_sec)} – {formatTime(selectedEvent.end_sec)}</span></div>
          <p>{selectedEvent.description}</p>
        </div>}
      </div>
      <aside className="events-list-pane">
        <div className="events-list-heading"><span>All events</span><span>{events.length}</span></div>
        <div className="events-list-scroll">
          {loading && <div className="events-list-state">Loading events…</div>}
          {error && <div className="events-list-state error">{error}</div>}
          {!loading && !error && events.length === 0 && <div className="events-list-state"><p>No events have been imported.</p><button onClick={onCheckJson}><FileJson2 size={14} />Check events.json</button></div>}
          {events.map((event) => <button className={`library-event ${selectedEvent?.event_id === event.event_id ? "selected" : ""}`} key={event.event_id} onClick={() => onSelect(event.event_id)}>
            <div className="library-event-top"><span>{(event.event_type || "event").replaceAll("_", " ")}</span><time>{formatTime(event.start_sec)}</time></div>
            <p>{event.description}</p>
            <div className="library-event-bottom"><span>{videos.find((video) => video.video_id === event.video_id)?.filename || event.video_id}</span><span>{event.anomaly_category || "Unassessed"}</span></div>
          </button>)}
        </div>
      </aside>
    </div>
  </section>;
}

function EventsReviewModal({ videos, file, preview, selectedVideoId, error, importing, mismatch, onFile, onSelectVideo, onClose, onImport }) {
  const canImport = Boolean(file && selectedVideoId && preview?.count > 0 && preview.invalidCount === 0 && !mismatch && !importing);
  return <div className="modal-scrim" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <section className="events-modal" role="dialog" aria-modal="true" aria-labelledby="events-modal-title">
      <header className="events-modal-header">
        <div><h2 id="events-modal-title">Check events.json</h2><p>Preview the file before importing.</p></div>
        <button className="icon-button modal-close" onClick={onClose} aria-label="Close"><X size={18} /></button>
      </header>
      <label className="events-file-picker">
        <input type="file" accept="application/json,.json" onChange={(event) => { const file = event.target.files?.[0]; event.target.value = ""; onFile(file); }} />
        <FileJson2 size={18} />
        <span><strong>{file?.name || "Choose events.json"}</strong><small>{file ? `${(file.size / 1024).toFixed(1)} KB` : "Select a Stage 4 JSON file to preview"}</small></span>
        <span className="file-picker-action">Browse</span>
      </label>
      {error && <div className="review-error">{error}</div>}
      {preview && <div className="events-preview">
        <div className="preview-summary">
          <div><strong>{preview.count}</strong><span>events</span></div>
          <div><strong>{preview.format}</strong><span>format</span></div>
          <div><strong>{preview.invalidCount === 0 ? "Valid" : `${preview.invalidCount} issues`}</strong><span>records</span></div>
        </div>
        {preview.sourceVideoIds.length > 0 && <div className="preview-source">Video ID in file: {preview.sourceVideoIds.join(", ")}</div>}
        {preview.invalidCount > 0 && <div className="review-error">Some records need an event type, description, and valid timestamps.</div>}
        {preview.count === 0 && <div className="preview-empty">No events found in this file.</div>}
        {preview.sample.length > 0 && <div className="preview-events">
          {preview.sample.map((event, index) => <article className="preview-event" key={event.event_id || index}>
            <div><strong>{String(event.event_type || event.type || "Event").replaceAll("_", " ")}</strong><span>{formatTime(event.start_sec ?? event.start)}–{formatTime(event.end_sec ?? event.end)}</span></div>
            <p>{event.description || "No description"}</p>
          </article>)}
          {preview.count > preview.sample.length && <div className="preview-more">and {preview.count - preview.sample.length} more</div>}
        </div>}
      </div>}
      <label className="target-video-label" htmlFor="events-target-video">Import into video</label>
      <select id="events-target-video" className="target-video-select" value={selectedVideoId} onChange={(event) => onSelectVideo(event.target.value)}>
        <option value="">Select the matching video</option>
        {videos.map((video) => <option key={video.video_id} value={video.video_id}>{video.filename}</option>)}
      </select>
      {mismatch && <div className="review-error">The file’s video ID does not match the selected video.</div>}
      {videos.length === 0 && <div className="preview-empty">Add a video before importing. You can still preview this file.</div>}
      <footer className="events-modal-footer">
        <button className="modal-cancel" onClick={onClose}>Close</button>
        <button className="modal-import" onClick={onImport} disabled={!canImport}>{importing ? <LoaderCircle size={15} className="spin" /> : <FileJson2 size={15} />}{importing ? "Importing…" : "Import events"}</button>
      </footer>
    </section>
  </div>;
}

function Message({ message, videos, onProcess, onImport, activeClip, onPlay }) {
  const video = videos.find((item) => item.video_id === message.videoId);
  if (message.role === "user") return <div className="user-row"><div className="user-bubble">{message.text}</div></div>;
  return <div className="assistant-row">
    <div className="assistant-avatar"><Activity size={15} /></div>
    <div className="assistant-content">
      {message.videoId && video && <VideoWorkflow video={video} onProcess={onProcess} onImport={onImport} />}
      {message.text && <p className="assistant-copy">{message.text}</p>}
      {message.result && <>
        <div className="result-meta"><span className="grounded-pill"><Check size={12} /> MATCHING EVENTS</span><span>{message.result.total_matches} records</span></div>
        {message.result.matched_events?.length > 0 && <div className="event-results">
          {message.result.matched_events.map((event) => <EventCard key={event.event_id} event={event} active={activeClip === event.event_id} onPlay={() => onPlay(activeClip === event.event_id ? null : event.event_id)} />)}
        </div>}
      </>}
    </div>
  </div>;
}

function VideoWorkflow({ video, onProcess, onImport }) {
  const processing = video.status === "processing";
  const waiting = video.status === "awaiting_events";
  const failed = video.status === "failed";
  return <div className="video-workflow-card">
    <div className="video-workflow-head"><div className="workflow-video-icon"><Clapperboard size={17} /></div><div className="workflow-video-info"><strong>{video.filename}</strong><span>{video.resolution} · {formatTime(video.duration_sec)} · {video.fps?.toFixed(1) || "—"} fps</span></div><span className={`video-state ${video.status}`}>{processing && <LoaderCircle size={12} className="spin" />}{waiting ? "READY FOR EVENTS" : processing ? "PROCESSING" : failed ? "NEEDS ATTENTION" : "ADDED"}</span></div>
    <div className="workflow-progress"><span className={processing ? "active" : waiting || video.status === "completed" ? "done" : ""}>1</span><i className={waiting || video.status === "completed" ? "done" : ""} /><span className={waiting ? "active" : video.status === "completed" ? "done" : ""}>2</span><i className={video.status === "completed" ? "done" : ""} /><span className={video.status === "completed" ? "active" : ""}>3</span><div><strong>Run tracking</strong><strong>Import descriptions</strong><strong>Search evidence</strong></div></div>
    {failed && <div className="workflow-error">{video.error_message || "Processing failed. You can try again."}</div>}
    <div className="workflow-footer">
      <span>{video.status === "completed" ? "Descriptions are imported and ready to search." : waiting ? "Tracking windows are ready. Import the final Stage 4 events.json." : processing ? "Tracking is running in the background. You can keep chatting." : "Run tracking, then import the final Stage 4 events.json."}</span>
      {waiting ? <button className="workflow-primary" onClick={onImport}><FileJson2 size={14} />Import events.json</button> : !processing && video.status !== "completed" && <button className="workflow-primary" onClick={() => onProcess(video.video_id)}><Activity size={14} />{failed ? "Try again" : "Run tracking"}</button>}
    </div>
  </div>;
}

function EventCard({ event, active, onPlay }) {
  return <article className="event-card">
    <div className="event-card-top"><span className="event-type"><span className="event-type-mark" />{(event.event_type || "event").replaceAll("_", " ")}</span><span className="event-confidence">{Math.round((event.confidence || 0) * 100)}% confidence</span></div>
    <p>{event.description}</p>
    <div className="event-card-footer"><span><Clock3 size={13} />{formatTime(event.start_sec)} – {formatTime(event.end_sec)}</span><span><Shield size={13} />{event.anomaly_category || "Unassessed"}</span>{event.entity_ids?.length > 0 && <span><Camera size={13} />Track {event.entity_ids.join(", ")}</span>}
      {event.clip_url && <button className="evidence-link" onClick={onPlay}>{active ? "Close clip" : "Play evidence"}<ArrowUpRight size={13} /></button>}
    </div>
    {active && event.clip_url && <video className="evidence-video" controls playsInline src={event.clip_url} />}
  </article>;
}
