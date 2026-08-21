const sessionList = document.querySelector("#session-list");
const sessionForm = document.querySelector("#session-form");
const sessionTitle = document.querySelector("#session-title");
const sessionError = document.querySelector("#session-error");
const sessionView = document.querySelector("#session-view");
const emptyState = document.querySelector("#empty-state");
const activeSessionTitle = document.querySelector("#active-session-title");
const sessionStatus = document.querySelector("#session-status");
const historyList = document.querySelector("#history-list");
const historyCount = document.querySelector("#history-count");
const researchForm = document.querySelector("#research-form");
const questionInput = document.querySelector("#question");
const researchButton = document.querySelector("#research-button");
const researchError = document.querySelector("#research-error");
const uploadForm = document.querySelector("#upload-form");
const uploadMessage = document.querySelector("#upload-message");
const metricsGrid = document.querySelector("#metrics-grid");
const evaluationPanel = document.querySelector("#evaluation-panel");
const evaluationResults = document.querySelector("#evaluation-results");
const evaluationCount = document.querySelector("#evaluation-count");
const evaluationMessage = document.querySelector("#evaluation-message");

let activeSessionId = null;

window.addEventListener("DOMContentLoaded", () => {
  loadSessions();
  loadMetrics();
});
document.querySelector("#new-session-button").addEventListener("click", showSessionForm);
document.querySelector("#empty-new-session-button").addEventListener("click", showSessionForm);
document.querySelector("#cancel-session-button").addEventListener("click", hideSessionForm);
sessionForm.addEventListener("submit", createSession);
researchForm.addEventListener("submit", askResearchQuestion);
uploadForm.addEventListener("submit", uploadDocument);
document.querySelector("#refresh-metrics-button").addEventListener("click", loadMetrics);
document.querySelector("#run-evaluation-button").addEventListener("click", runEvaluation);

async function loadSessions() {
  clearError(sessionError);
  try {
    const data = await requestWithRetry("/sessions");
    renderSessionList(data.sessions);
    if (data.sessions.length > 0) {
      await openSession(data.sessions[0].session_id);
    } else {
      showEmptyState();
    }
  } catch (error) {
    showError(sessionError, "Could not load research sessions. Please refresh when the API is running.");
  }
}

function showSessionForm() {
  sessionForm.hidden = false;
  sessionTitle.focus();
}

function hideSessionForm() {
  sessionForm.hidden = true;
  sessionTitle.value = "";
}

async function createSession(event) {
  event.preventDefault();
  clearError(sessionError);
  try {
    const created = await request("/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: sessionTitle.value.trim() }),
    });
    hideSessionForm();
    await loadSessions();
    await openSession(created.session_id);
  } catch (error) {
    showError(sessionError, error.message);
  }
}

function renderSessionList(sessions) {
  sessionList.replaceChildren(...sessions.map((session) => {
    const button = document.createElement("button");
    button.className = "session-item";
    button.type = "button";
    button.dataset.sessionId = session.session_id;
    button.textContent = session.title;
    button.addEventListener("click", () => openSession(session.session_id));
    return button;
  }));
}

async function openSession(sessionId) {
  clearError(sessionError);
  try {
    const session = await request(`/sessions/${encodeURIComponent(sessionId)}`);
    activeSessionId = session.session_id;
    activeSessionTitle.textContent = session.title;
    sessionView.hidden = false;
    emptyState.hidden = true;
    document.querySelectorAll(".session-item").forEach((item) => {
      item.classList.toggle("active", item.dataset.sessionId === sessionId);
    });
    renderHistory(session.questions);
  } catch (error) {
    showError(sessionError, error.message);
  }
}

function showEmptyState() {
  activeSessionId = null;
  sessionView.hidden = true;
  emptyState.hidden = false;
  historyList.replaceChildren();
}

function renderHistory(results) {
  historyCount.textContent = `${results.length} ${results.length === 1 ? "result" : "results"}`;
  if (!results.length) {
    historyList.innerHTML = '<div class="no-history">No research yet. Ask the first question for this session.</div>';
    return;
  }
  historyList.replaceChildren(...results.map(renderHistoryEntry));
}

function renderHistoryEntry(result) {
  const entry = document.createElement("article");
  entry.className = "history-entry";

  const question = document.createElement("h3");
  question.className = "history-question";
  question.textContent = result.question;
  entry.appendChild(question);

  const answer = document.createElement("p");
  answer.className = "history-answer";
  answer.textContent = result.answer;
  entry.appendChild(answer);

  const metadata = document.createElement("div");
  metadata.className = "metadata";
  metadata.appendChild(metadataBadge(`Local documents: ${result.used_local_documents ? "Yes" : "No"}`));
  metadata.appendChild(metadataBadge(`Web search: ${result.used_web_search ? "Yes" : "No"}`));
  entry.appendChild(metadata);

  const sources = document.createElement("div");
  sources.className = "sources";
  sources.replaceChildren(...result.sources.map(renderSource));
  entry.appendChild(sources);

  const timestamp = document.createElement("div");
  timestamp.className = "history-time";
  timestamp.textContent = formatDate(result.created_at);
  entry.appendChild(timestamp);
  return entry;
}

function metadataBadge(text) {
  const badge = document.createElement("span");
  badge.textContent = text;
  return badge;
}

function renderSource(source) {
  const item = document.createElement("div");
  item.className = "source";
  const name = document.createElement("div");
  name.className = "source-name";
  name.textContent = `${source.source_type === "web" ? "🌐" : "📄"} ${source.title || source.source}`;
  item.appendChild(name);

  const meta = document.createElement("div");
  meta.className = "source-meta";
  const score = source.score === null || source.score === undefined
    ? ""
    : ` · relevance ${Number(source.score).toFixed(3)}`;
  meta.textContent = `${source.source_type === "web" ? "Web source" : "Local document"}${score}`;
  item.appendChild(meta);

  if (source.url) {
    const link = document.createElement("a");
    link.href = source.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = source.url;
    item.appendChild(link);
  }
  return item;
}

async function askResearchQuestion(event) {
  event.preventDefault();
  if (!activeSessionId) return showError(researchError, "Create a session before researching.");
  const question = questionInput.value.trim();
  if (!question) return showError(researchError, "Enter a research question.");

  setResearching(true);
  try {
    await request(`/sessions/${encodeURIComponent(activeSessionId)}/research`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    questionInput.value = "";
    await openSession(activeSessionId);
    sessionStatus.textContent = "Research complete.";
    sessionStatus.classList.add("complete");
  } catch (error) {
    showError(researchError, error.message);
    sessionStatus.textContent = "Research failed";
    sessionStatus.classList.remove("complete");
  } finally {
    setResearching(false);
  }
}

function setResearching(isResearching) {
  researchButton.disabled = isResearching;
  questionInput.disabled = isResearching;
  sessionStatus.textContent = isResearching ? "Researching..." : "Ready";
  sessionStatus.classList.toggle("complete", false);
  if (!isResearching) clearError(researchError);
}

async function uploadDocument(event) {
  event.preventDefault();
  const fileInput = document.querySelector("#document-file");
  if (!fileInput.files.length) return;
  uploadMessage.hidden = false;
  uploadMessage.textContent = "Indexing document...";
  try {
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    const data = await request("/documents/upload", { method: "POST", body: formData });
    uploadMessage.textContent = `${data.filename}: ${data.status} (${data.chunks_created} chunks)`;
    fileInput.value = "";
  } catch (error) {
    uploadMessage.textContent = error.message;
  }
}

async function loadMetrics() {
  try {
    const metrics = await request("/metrics");
    const values = [
      ["Requests", metrics.total_requests],
      ["Successful", metrics.successful_requests],
      ["Avg latency", `${Number(metrics.average_latency_ms).toFixed(1)} ms`],
      ["Avg retrieval", `${Number(metrics.average_retrieval_latency_ms).toFixed(1)} ms`],
      ["Avg generation", `${Number(metrics.average_generation_latency_ms).toFixed(1)} ms`],
      ["Avg sources", Number(metrics.average_sources).toFixed(1)],
    ];
    metricsGrid.replaceChildren(...values.map(([label, value]) => {
      const item = document.createElement("div");
      item.className = "metric-item";
      item.innerHTML = `<strong>${value}</strong><span>${label}</span>`;
      return item;
    }));
  } catch (error) {
    metricsGrid.textContent = error.message;
  }
}

async function runEvaluation() {
  evaluationMessage.hidden = false;
  evaluationMessage.textContent = "Evaluation running...";
  const button = document.querySelector("#run-evaluation-button");
  button.disabled = true;
  try {
    const report = await request("/evaluation/run", { method: "POST" });
    evaluationCount.textContent = `${report.total_questions} questions`;
    const values = [
      ["Retrieval hit rate", `${(report.retrieval_hit_rate * 100).toFixed(1)}%`],
      ["Source match rate", `${(report.source_match_rate * 100).toFixed(1)}%`],
      ["Keyword coverage", `${(report.keyword_coverage * 100).toFixed(1)}%`],
      ["Average latency", `${Number(report.average_latency_ms).toFixed(1)} ms`],
    ];
    evaluationResults.replaceChildren(...values.map(([label, value]) => {
      const item = document.createElement("div");
      item.className = "evaluation-item";
      item.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
      return item;
    }));
    evaluationPanel.hidden = false;
    evaluationMessage.textContent = "Evaluation complete.";
  } catch (error) {
    evaluationMessage.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || "The server could not complete the request.");
  return data;
}

async function requestWithRetry(url, options = {}, attempts = 3) {
  let lastError;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      return await request(url, options);
    } catch (error) {
      lastError = error;
      if (attempt < attempts - 1) {
        await new Promise((resolve) => setTimeout(resolve, 250 * (attempt + 1)));
      }
    }
  }
  throw lastError;
}

function showError(element, message) {
  element.textContent = message;
  element.hidden = false;
}

function clearError(element) {
  element.textContent = "";
  element.hidden = true;
}

function formatDate(value) {
  return new Date(value).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}
