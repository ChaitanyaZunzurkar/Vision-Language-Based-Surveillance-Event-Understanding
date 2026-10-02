/**
 * SurveillanceLens — Frontend Client Application Logic
 */

const API_BASE = window.location.origin;

// State management
const state = {
  currentVideo: null,
  videos: [],
  events: [],
  selectedFile: null,
  pollingInterval: null,
  currentFilter: "",
};

// DOM Elements
const elements = {
  // Metrics
  statVideos: document.getElementById("stat-total-videos"),
  statEvents: document.getElementById("stat-total-events"),
  statAnomalies: document.getElementById("stat-total-anomalies"),
  systemStatus: document.getElementById("system-status-indicator"),

  // Upload
  dropzone: document.getElementById("video-dropzone"),
  fileInput: document.getElementById("video-file-input"),
  fileInfoBar: document.getElementById("file-info-bar"),
  fileName: document.getElementById("selected-file-name"),
  fileSize: document.getElementById("selected-file-size"),
  btnUpload: document.getElementById("btn-upload"),
  progressContainer: document.getElementById("upload-progress-container"),
  progressBar: document.getElementById("upload-progress-bar"),
  progressText: document.getElementById("upload-progress-text"),

  // Player & Metadata
  player: document.getElementById("main-video-player"),
  playerPlaceholder: document.getElementById("player-placeholder"),
  activeVideoStatus: document.getElementById("active-video-status"),
  videoMetaStrip: document.getElementById("video-meta-strip"),
  metaDuration: document.getElementById("meta-duration"),
  metaFps: document.getElementById("meta-fps"),
  metaResolution: document.getElementById("meta-resolution"),
  metaAnomaly: document.getElementById("meta-anomaly"),
  btnRunPipeline: document.getElementById("btn-run-pipeline"),
  pipelineTracker: document.getElementById("pipeline-tracker"),

  // Video Archive
  videoArchiveList: document.getElementById("video-archive-list"),
  btnRefreshVideos: document.getElementById("btn-refresh-videos"),
  stage4JsonInput: document.getElementById("stage4-json-input"),
  btnImportStage4: document.getElementById("btn-import-stage4"),
  stage4ImportStatus: document.getElementById("stage4-import-status"),
  workflowSelectedVideo: document.getElementById("workflow-selected-video"),
  pipelineStatusTitle: document.getElementById("pipeline-status-title"),
  pipelineStatusMessage: document.getElementById("pipeline-status-message"),
  pipelineSpinner: document.getElementById("pipeline-spinner"),

  // Events & Feed
  eventsFeed: document.getElementById("events-feed"),
  eventsEmptyState: document.getElementById("events-empty-state"),
  eventFeedCount: document.getElementById("event-feed-count"),
  eventTypeFilter: document.getElementById("event-type-filter"),

  // NL Query
  queryForm: document.getElementById("nl-query-form"),
  queryInput: document.getElementById("nl-query-input"),
  btnQuerySearch: document.getElementById("btn-query-search"),
  ragSummaryBox: document.getElementById("rag-summary-box"),
  ragSummaryText: document.getElementById("rag-summary-text"),
  ragMatchCount: document.getElementById("rag-match-count"),

  // Modal
  clipModal: document.getElementById("clip-modal"),
  modalClipPlayer: document.getElementById("modal-clip-player"),
  modalClipTitle: document.getElementById("modal-clip-title"),
  modalClipDesc: document.getElementById("modal-clip-desc"),
  modalClipTime: document.getElementById("modal-clip-time"),
  modalClipEventType: document.getElementById("modal-clip-event-type"),
  modalClipAnomaly: document.getElementById("modal-clip-anomaly"),
  modalClipEntities: document.getElementById("modal-clip-entities"),
  btnCloseModal: document.getElementById("btn-close-modal"),
};

// ==========================================================================
// Initialization
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  fetchSystemStats();
  fetchVideoArchive();
});

function initEventListeners() {
  // Drag & drop upload
  elements.dropzone.addEventListener("click", () => elements.fileInput.click());
  elements.fileInput.addEventListener("change", handleFileSelected);

  elements.dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    elements.dropzone.classList.add("dragover");
  });

  elements.dropzone.addEventListener("dragleave", () => {
    elements.dropzone.classList.remove("dragover");
  });

  elements.dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    elements.dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected({ target: { files: e.dataTransfer.files } });
    }
  });

  elements.btnUpload.addEventListener("click", uploadSelectedVideo);
  elements.btnRunPipeline.addEventListener("click", () => {
    if (state.currentVideo) {
      triggerPipeline(state.currentVideo.video_id);
    }
  });

  elements.btnRefreshVideos.addEventListener("click", fetchVideoArchive);
  elements.btnImportStage4.addEventListener("click", importStage4Events);
  elements.stage4JsonInput.addEventListener("change", () => {
    const file = elements.stage4JsonInput.files[0];
    elements.btnImportStage4.disabled = !state.currentVideo || !file;
    elements.stage4ImportStatus.textContent = file
      ? `Selected: ${file.name}. This must be the final Stage 4 events.json.`
      : "Waiting for a video and events.json.";
  });

  // Filters & Query
  elements.eventTypeFilter.addEventListener("change", (e) => {
    state.currentFilter = e.target.value;
    renderEvents();
  });

  elements.queryForm.addEventListener("submit", handleNaturalLanguageQuery);

  // Modal close
  elements.btnCloseModal.addEventListener("click", closeModal);
  elements.clipModal.addEventListener("click", (e) => {
    if (e.target === elements.clipModal) closeModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });
}

// ==========================================================================
// Video Upload & File Selection
// ==========================================================================

function handleFileSelected(event) {
  const file = event.target.files[0];
  if (!file) return;

  state.selectedFile = file;
  elements.fileName.textContent = file.name;
  elements.fileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
  elements.fileInfoBar.classList.remove("hidden");
}

async function uploadSelectedVideo() {
  if (!state.selectedFile) return;

  const formData = new FormData();
  formData.append("file", state.selectedFile);
  formData.append("auto_run", "false");

  elements.btnUpload.disabled = true;
  elements.progressContainer.classList.remove("hidden");
  elements.progressBar.style.width = "40%";
  elements.progressText.textContent = "Adding source video to the archive…";

  try {
    const response = await fetch(`${API_BASE}/api/videos/upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Failed to upload video");
    }

    elements.progressBar.style.width = "100%";
    elements.progressText.textContent = "Upload complete!";

    const videoRecord = await response.json();

    // Reset upload state
    setTimeout(() => {
      elements.progressContainer.classList.add("hidden");
      elements.fileInfoBar.classList.add("hidden");
      elements.fileInput.value = "";
      state.selectedFile = null;
      elements.btnUpload.disabled = false;
    }, 1200);

    // Refresh archive & set active
    await fetchVideoArchive();
    selectVideo(videoRecord);

    elements.workflowSelectedVideo.textContent = `Selected source video: ${videoRecord.filename}. Now choose Path A or Path B below.`;
  } catch (error) {
    alert(`Upload error: ${error.message}`);
    elements.btnUpload.disabled = false;
    elements.progressContainer.classList.add("hidden");
  }
}

// ==========================================================================
// Video Selection & Playback
// ==========================================================================

function selectVideo(video) {
  const selectionChanged = state.currentVideo?.video_id !== video.video_id;
  if (selectionChanged) {
    elements.stage4JsonInput.value = "";
    elements.stage4ImportStatus.textContent = "Choose the final Stage 4 events.json for this video.";
    elements.pipelineTracker.classList.add("hidden");
  }
  state.currentVideo = video;
  elements.workflowSelectedVideo.textContent = `Selected source video: ${video.filename}. Now choose Path A or Path B below.`;
  elements.btnImportStage4.disabled = !elements.stage4JsonInput.files.length;

  // Update UI Elements
  elements.playerPlaceholder.classList.add("hidden");
  elements.player.src = `${API_BASE}${video.stream_url}`;
  elements.player.load();

  // Status Badge
  elements.activeVideoStatus.textContent = video.status.toUpperCase();
  elements.activeVideoStatus.className = `badge ${video.status}`;

  // Metadata Strip
  elements.metaDuration.textContent = `${video.duration_sec.toFixed(1)}s`;
  elements.metaFps.textContent = video.fps.toFixed(1);
  elements.metaResolution.textContent = video.resolution;
  elements.metaAnomaly.textContent = "Not assessed";
  elements.videoMetaStrip.classList.remove("hidden");

  // Run Button state
  elements.btnRunPipeline.disabled = video.status === "processing";

  // Highlight in list
  document.querySelectorAll(".video-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.videoId === video.video_id);
  });

  // Load events for this video
  fetchEventsForVideo(video.video_id);

  if (video.status === "processing") {
    startPipelineProgressTracking(video.video_id);
  }
}

// ==========================================================================
// Pipeline Execution & Status Tracking
// ==========================================================================

async function triggerPipeline(videoId) {
  elements.btnRunPipeline.disabled = true;
  elements.btnRunPipeline.textContent = "Starting local processing…";
  startPipelineProgressTracking(videoId);

  try {
    const res = await fetch(`${API_BASE}/api/pipeline/run/${videoId}?run_in_background=true`, {
      method: "POST",
    });
    if (!res.ok) throw new Error("Failed to trigger pipeline");
  } catch (e) {
    alert(`Pipeline execution error: ${e.message}`);
    stopPipelineProgressTracking();
  } finally {
    elements.btnRunPipeline.textContent = "Run local tracking + windows";
  }
}

function startPipelineProgressTracking(videoId) {
  elements.pipelineTracker.classList.remove("hidden");
  elements.pipelineStatusTitle.textContent = "YOLO + ByteTrack + Stage 2 running";
  elements.pipelineStatusMessage.textContent = "This can take several minutes. The app is processing the video locally; keep this page open.";
  elements.pipelineSpinner.classList.remove("hidden");

  if (state.pollingInterval) clearInterval(state.pollingInterval);

  state.pollingInterval = setInterval(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/pipeline/status/${videoId}`);
      if (!res.ok) return;

      const data = await res.json();
      if (["completed", "awaiting_events", "failed"].includes(data.status)) {
        clearInterval(state.pollingInterval);
        state.pollingInterval = null;
        elements.pipelineSpinner.classList.add("hidden");
        elements.btnRunPipeline.disabled = false;
        elements.pipelineStatusTitle.textContent = data.status === "awaiting_events" ? "Stages 1b–2 complete" : data.status === "failed" ? "Processing failed" : "Complete";
        elements.pipelineStatusMessage.textContent = data.status === "awaiting_events"
          ? "tracks.csv and windows.json are ready. Run the Kaggle Stage 4 notebook with this video and windows.json, then import its events.json in Path B."
          : data.status === "failed"
            ? (data.error_message || "Check the server terminal for details.")
            : `Processing finished with ${data.total_events} event(s).`;

        // Refresh details
        const videoRes = await fetch(`${API_BASE}/api/videos/${videoId}`);
        if (videoRes.ok) {
          const updated = await videoRes.json();
          selectVideo(updated);
        }
        fetchSystemStats();
        fetchVideoArchive();
      }
    } catch (err) {
      console.error("Status polling failed:", err);
    }
  }, 1500);
}

function stopPipelineProgressTracking() {
  if (state.pollingInterval) {
    clearInterval(state.pollingInterval);
    state.pollingInterval = null;
  }
  elements.pipelineTracker.classList.add("hidden");
  elements.pipelineSpinner.classList.add("hidden");
  elements.btnRunPipeline.disabled = false;
  elements.btnRunPipeline.textContent = "Run local tracking + windows";
}

// ==========================================================================
// Event Feed & Semantic Descriptions
// ==========================================================================

async function fetchEventsForVideo(videoId) {
  try {
    const res = await fetch(`${API_BASE}/api/events?video_id=${videoId}`);
    if (!res.ok) return;

    state.events = await res.json();
    renderEvents();
  } catch (err) {
    console.error("Failed to fetch events:", err);
  }
}

async function importStage4Events() {
  if (!state.currentVideo) {
    elements.stage4ImportStatus.textContent = "Add or select the matching source video above first.";
    return;
  }
  const file = elements.stage4JsonInput.files[0];
  if (!file) {
    elements.stage4ImportStatus.textContent = "Choose the final Stage 4 events.json file first.";
    return;
  }

  const formData = new FormData();
  formData.append("video_id", state.currentVideo.video_id);
  formData.append("file", file);
  elements.btnImportStage4.disabled = true;
  elements.stage4ImportStatus.textContent = "Importing events and creating evidence clips…";
  try {
    const response = await fetch(`${API_BASE}/api/events/import`, { method: "POST", body: formData });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Event import failed");
    elements.stage4ImportStatus.textContent = `Imported ${result.imported_events} events for ${state.currentVideo.filename}.`;
    await fetchEventsForVideo(state.currentVideo.video_id);
    await fetchSystemStats();
    const selectedId = state.currentVideo.video_id;
    const updated = await fetch(`${API_BASE}/api/videos/${selectedId}`);
    if (updated.ok) selectVideo(await updated.json());
    await fetchVideoArchive();
  } catch (error) {
    elements.stage4ImportStatus.textContent = error.message;
  } finally {
    elements.btnImportStage4.disabled = false;
  }
}

function renderEvents() {
  const feed = elements.eventsFeed;
  feed.innerHTML = "";

  const filtered = state.events.filter((ev) => {
    if (!state.currentFilter) return true;
    return ev.event_type.toLowerCase() === state.currentFilter.toLowerCase();
  });

  elements.eventFeedCount.textContent = filtered.length;

  if (filtered.length === 0) {
    elements.eventsEmptyState.classList.remove("hidden");
    feed.appendChild(elements.eventsEmptyState);
    return;
  }

  elements.eventsEmptyState.classList.add("hidden");

  // Determine anomaly status for active video metadata
  const hasAnomaly = filtered.some((e) => e.anomaly_category && !["Normal", "Unassessed"].includes(e.anomaly_category));
  if (hasAnomaly) {
    const topAnom = filtered.find((e) => !["Normal", "Unassessed"].includes(e.anomaly_category));
    elements.metaAnomaly.textContent = topAnom.anomaly_category;
    elements.metaAnomaly.className = "meta-val alert-tag is-anom";
  } else {
    elements.metaAnomaly.textContent = filtered.some((e) => e.anomaly_category === "Unassessed") ? "Not assessed" : "Normal";
    elements.metaAnomaly.className = "meta-val alert-tag";
  }

  filtered.forEach((ev) => {
    const card = document.createElement("div");
    card.className = "event-card";
    card.dataset.type = ev.event_type.toLowerCase();

    const isAnomalous = ev.anomaly_category && !["Normal", "Unassessed"].includes(ev.anomaly_category);

    card.innerHTML = `
      <div class="event-card-top">
        <div class="event-tags-wrap">
          <span class="event-type-pill">${ev.event_type}</span>
          <span class="anomaly-pill ${isAnomalous ? "is-anom" : ""}">${ev.anomaly_category}</span>
        </div>
        <div class="event-time-span">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
          </svg>
          ${ev.start_sec.toFixed(1)}s - ${ev.end_sec.toFixed(1)}s
        </div>
      </div>

      <p class="event-description">${escapeHtml(ev.description)}</p>

      <div class="event-card-bottom">
        <div class="event-entities">
          ${(ev.entity_ids || [])
            .map((id) => `<span class="entity-chip">Entity ID #${id}</span>`)
            .join("")}
          <span class="entity-chip">${Math.round(ev.confidence * 100)}% conf</span>
        </div>

        ${
          ev.clip_url
            ? `<button class="btn-watch-clip" data-event-id="${ev.event_id}">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
                Watch Evidence
              </button>`
            : ""
        }
      </div>
    `;

    // Watch evidence clip handler
    const clipBtn = card.querySelector(".btn-watch-clip");
    if (clipBtn) {
      clipBtn.addEventListener("click", () => openClipModal(ev));
    }

    feed.appendChild(card);
  });
}

// ==========================================================================
// Natural Language Query & Retrieval
// ==========================================================================

async function handleNaturalLanguageQuery(e) {
  e.preventDefault();
  const query = elements.queryInput.value.trim();
  if (!query) return;

  elements.btnQuerySearch.disabled = true;
  elements.btnQuerySearch.textContent = "Searching...";

  try {
    const res = await fetch(`${API_BASE}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 5 }),
    });

    if (!res.ok) throw new Error("Search query failed");

    const data = await res.json();

    // Render Grounded RAG Summary
    elements.ragSummaryBox.classList.remove("hidden");
    elements.ragMatchCount.textContent = `${data.total_matches} event(s) matched`;
    elements.ragSummaryText.textContent = data.grounded_summary;

    // If query matches events, render them in feed
    if (data.matched_events && data.matched_events.length > 0) {
      state.events = data.matched_events;
      renderEvents();
    }
  } catch (err) {
    alert(`Query failed: ${err.message}`);
  } finally {
    elements.btnQuerySearch.disabled = false;
    elements.btnQuerySearch.textContent = "Search";
  }
}

// ==========================================================================
// Evidence Clip Verification Modal
// ==========================================================================

function openClipModal(ev) {
  elements.modalClipPlayer.src = `${API_BASE}${ev.clip_url}`;
  elements.modalClipPlayer.load();

  elements.modalClipTitle.textContent = `Evidence Clip: ${ev.event_type.toUpperCase()}`;
  elements.modalClipDesc.textContent = ev.description;
  elements.modalClipTime.textContent = `${ev.start_sec.toFixed(1)}s - ${ev.end_sec.toFixed(1)}s`;
  elements.modalClipEventType.textContent = ev.event_type.toUpperCase();
  elements.modalClipAnomaly.textContent = ev.anomaly_category;
  elements.modalClipEntities.textContent = (ev.entity_ids || []).map((id) => `#${id}`).join(", ") || "None";

  elements.clipModal.classList.remove("hidden");
  elements.modalClipPlayer.play().catch(() => {});
}

function closeModal() {
  elements.modalClipPlayer.pause();
  elements.modalClipPlayer.src = "";
  elements.clipModal.classList.add("hidden");
}

// ==========================================================================
// Footage Archive & Metrics
// ==========================================================================

async function fetchVideoArchive() {
  try {
    const res = await fetch(`${API_BASE}/api/videos`);
    if (!res.ok) return;

    state.videos = await res.json();
    renderVideoArchive();

    // Select first video if none selected
    if (!state.currentVideo && state.videos.length > 0) {
      selectVideo(state.videos[0]);
    }
  } catch (err) {
    console.error("Failed to fetch archive:", err);
  }
}

function renderVideoArchive() {
  const list = elements.videoArchiveList;
  list.innerHTML = "";

  if (state.videos.length === 0) {
    list.innerHTML = '<div class="empty-state">No videos ingested yet. Upload one above!</div>';
    return;
  }

  state.videos.forEach((v) => {
    const item = document.createElement("div");
    item.className = `video-item ${state.currentVideo && state.currentVideo.video_id === v.video_id ? "active" : ""}`;
    item.dataset.videoId = v.video_id;

    item.innerHTML = `
      <div class="video-item-left">
        <svg class="video-item-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect width="18" height="18" x="3" y="3" rx="2"/>
          <path d="m9 8 6 4-6 4Z"/>
        </svg>
        <div>
          <div class="video-item-name" title="${escapeHtml(v.filename)}">${escapeHtml(v.filename)}</div>
          <div class="video-item-meta">${v.duration_sec.toFixed(1)}s • ${v.resolution}</div>
        </div>
      </div>
      <span class="badge ${v.status}">${v.status}</span>
    `;

    item.addEventListener("click", () => selectVideo(v));
    list.appendChild(item);
  });
}

async function fetchSystemStats() {
  try {
    const res = await fetch(`${API_BASE}/api/stats/dashboard`);
    if (!res.ok) return;

    const data = await res.json();
    elements.statVideos.textContent = data.total_videos || 0;
    elements.statEvents.textContent = data.total_events || 0;

    const anomalySum = Object.values(data.anomaly_distribution || {}).reduce((a, b) => a + b, 0);
    elements.statAnomalies.textContent = anomalySum;
  } catch (err) {
    console.error("Failed to fetch stats:", err);
  }
}

function escapeHtml(text) {
  if (!text) return "";
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
