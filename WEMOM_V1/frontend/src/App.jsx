import React, { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const initialControls = {
  density: 0,
  pitch: 0,
  velocity: 0,
  scale: "Auto",
  grid: "Auto",
  rule_vel: "Auto",
};

export default function App() {
  const [sessionId, setSessionId] = useState("");
  const [diaryId, setDiaryId] = useState(null);
  const [text, setText] = useState("");
  const [lastSubmittedIndex, setLastSubmittedIndex] = useState(0);
  const [controls, setControls] = useState(initialControls);
  const [entries, setEntries] = useState([]);
  const [queue, setQueue] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState("");
  const [mergeState, setMergeState] = useState({
    status: "idle",
    wavUrl: "",
    evaluation: null,
  });

  useEffect(() => {
    const startDiary = async () => {
      try {
        const res = await fetch(`${API_BASE}/diary/start`, { method: "POST" });
        const data = await res.json();
        setSessionId(data.session_id);
        setDiaryId(data.diary_id);
      } catch (err) {
        setError("Failed to start diary session.");
      }
    };

    startDiary();
  }, []);

  const apiUrl = useMemo(() => new URL(API_BASE), []);

  const resolveAudioUrl = (path) => {
    if (!path) return "";
    if (path.startsWith("http")) return path;
    return new URL(path, apiUrl).toString();
  };

  const enqueueSentence = () => {
    if (!sessionId) return;
    const newText = text.slice(lastSubmittedIndex).trim();
    if (!newText) return;
    const entryId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const queuedText = newText;
    const queuedControls = { ...controls };

    setEntries((prev) => [
      ...prev,
      {
        localId: entryId,
        text: queuedText,
        status: "queued",
        wavUrl: "",
        evaluation: null,
      },
    ]);
    setQueue((prev) => [...prev, { localId: entryId, text: queuedText, controls: queuedControls }]);

    const nextText = text.endsWith("\n") ? text : `${text}\n`;
    setText(nextText);
    setLastSubmittedIndex(nextText.length);
  };

  const processQueueItem = async (item) => {
    setEntries((prev) =>
      prev.map((entry) =>
        entry.localId === item.localId ? { ...entry, status: "processing" } : entry
      )
    );

    const payload = {
      session_id: sessionId,
      text: item.text,
      user_config: item.controls,
    };

    try {
      const res = await fetch(`${API_BASE}/diary/append`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error("append failed");
      }

      const data = await res.json();
      const wavUrl = resolveAudioUrl(data.wav_url);
      setEntries((prev) =>
        prev.map((entry) =>
          entry.localId === item.localId
            ? {
                ...entry,
                status: "done",
                wavUrl,
                evaluation: data.evaluation || null,
              }
            : entry
        )
      );
    } catch (err) {
      setEntries((prev) =>
        prev.map((entry) =>
          entry.localId === item.localId ? { ...entry, status: "error" } : entry
        )
      );
      setError("Generation failed. Please try again.");
    }
  };

  useEffect(() => {
    if (!sessionId || processing || queue.length === 0) return;
    const nextItem = queue[0];
    setProcessing(true);
    processQueueItem(nextItem).finally(() => {
      setQueue((prev) => prev.slice(1));
      setProcessing(false);
    });
  }, [queue, processing, sessionId]);

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      enqueueSentence();
    }
  };

  const handleMerge = async () => {
    if (!sessionId || processing) return;
    setMergeState({ status: "processing", wavUrl: "", evaluation: null });
    setError("");
    try {
      const res = await fetch(`${API_BASE}/diary/merge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });

      if (!res.ok) {
        throw new Error("merge failed");
      }

      const data = await res.json();
      const wavUrl = resolveAudioUrl(data.wav_url);
      setMergeState({ status: "done", wavUrl, evaluation: data.evaluation || null });
    } catch (err) {
      setMergeState({ status: "error", wavUrl: "", evaluation: null });
      setError("Diary merge failed. Please try again.");
    }
  };

  const handleTextChange = (event) => {
    const nextValue = event.target.value;
    const lockedPrefix = text.slice(0, lastSubmittedIndex);

    if (!nextValue.startsWith(lockedPrefix)) {
      return;
    }

    setText(nextValue);
  };

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>WEMOM Diary</h1>
          <p className="subtitle">
            Type a sentence, press Enter, and get a music snippet.
          </p>
        </div>
        <div className="badge">Diary {diaryId ?? "-"}</div>
      </header>

      <div className="content">
        <section className="main">
          <div className="input-card">
            <label htmlFor="sentence">Sentence</label>
            <textarea
              id="sentence"
              value={text}
              onChange={handleTextChange}
              onKeyDown={handleKeyDown}
              placeholder="Write one sentence and press Enter..."
              rows={6}
            />
            <div className="input-actions">
              <button onClick={enqueueSentence} disabled={!text.trim() || !sessionId}>
                {processing ? "Queueing..." : "Generate"}
              </button>
              <button onClick={handleMerge} disabled={!entries.length || processing}>
                {mergeState.status === "processing" ? "Finishing..." : "Finish Diary"}
              </button>
              <span className="hint">Enter to send, Shift+Enter for new line</span>
            </div>
          </div>

          {error && <div className="error">{error}</div>}

          <div className="entries">
            {entries.length === 0 ? (
              <div className="empty">No entries yet.</div>
            ) : (
              entries.map((entry) => (
                <div key={entry.localId} className="entry">
                  <div className="entry-text">{entry.text}</div>
                  <div className={`status status-${entry.status}`}>
                    {entry.status === "queued" && "Queued"}
                    {entry.status === "processing" && "Generating..."}
                    {entry.status === "done" && "Ready"}
                    {entry.status === "error" && "Error"}
                  </div>
                  {entry.wavUrl && <audio controls src={entry.wavUrl} />}
                  {entry.evaluation && <EvaluationPanel evaluation={entry.evaluation} />}
                </div>
              ))
            )}
          </div>

          {mergeState.status === "done" && (
            <div className="merge-result">
              <div className="merge-title">Diary Summary</div>
              {mergeState.wavUrl && <audio controls src={mergeState.wavUrl} />}
              {mergeState.evaluation && (
                <MacroEvaluationPanel evaluation={mergeState.evaluation} />
              )}
            </div>
          )}
        </section>

        <aside className="sidebar">
          <div className="panel">
            <h2>Latent Controls</h2>
            <Control
              label="Density"
              value={controls.density}
              min={-5}
              max={5}
              step={0.5}
              onChange={(v) => setControls({ ...controls, density: v })}
            />
            <Control
              label="Pitch"
              value={controls.pitch}
              min={-5}
              max={5}
              step={0.5}
              onChange={(v) => setControls({ ...controls, pitch: v })}
            />
            <Control
              label="Velocity"
              value={controls.velocity}
              min={-5}
              max={5}
              step={0.5}
              onChange={(v) => setControls({ ...controls, velocity: v })}
            />
          </div>

          <div className="panel">
            <h2>Rule Controls</h2>
            <SelectControl
              label="Scale"
              value={controls.scale}
              options={["Auto", "C Major", "C Minor", "Original"]}
              onChange={(v) => setControls({ ...controls, scale: v })}
            />
            <SelectControl
              label="Grid"
              value={controls.grid}
              options={["Auto", "1.0 (Quarter)", "0.5 (Eighth)", "0.25 (Sixteenth)"]}
              onChange={(v) => setControls({ ...controls, grid: v })}
            />
            <SelectControl
              label="Velocity Base"
              value={controls.rule_vel}
              options={["Auto", "60 (Soft)", "80 (Medium)", "100 (Hard)"]}
              onChange={(v) => setControls({ ...controls, rule_vel: v })}
            />
          </div>
        </aside>
      </div>
    </div>
  );
}

function EvaluationPanel({ evaluation }) {
  const percent = (value) =>
    typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "-";
  const number = (value, digits = 2) =>
    typeof value === "number" ? value.toFixed(digits) : "-";

  return (
    <div className="evaluation">
      <div>Notes: {evaluation.notes ?? "-"}</div>
      <div>Density: {number(evaluation.density)}</div>
      <div>Duration: {number(evaluation.duration)} s</div>
      <div>Velocity: {number(evaluation.velocity, 1)}</div>
      <div>C Maj: {percent(evaluation.c_maj_ratio)}</div>
      <div>C Min: {percent(evaluation.c_min_ratio)}</div>
    </div>
  );
}

function MacroEvaluationPanel({ evaluation }) {
  const number = (value, digits = 2) =>
    typeof value === "number" ? value.toFixed(digits) : "-";

  return (
    <div className="evaluation macro">
      <div>Pitch Var: {number(evaluation.traj_pitch_var)}</div>
      <div>Density Var: {number(evaluation.traj_density_var)}</div>
      <div>Avg Leap: {number(evaluation.smoothness_avg_leap)}</div>
      <div>Tonal Cohesion: {number(evaluation.tonal_cohesion_var)}</div>
      {evaluation.merged_eval && (
        <div className="macro-merged">
          <div>Notes: {evaluation.merged_eval.notes ?? "-"}</div>
          <div>Density: {number(evaluation.merged_eval.density)}</div>
          <div>Duration: {number(evaluation.merged_eval.duration)} s</div>
          <div>Velocity: {number(evaluation.merged_eval.velocity, 1)}</div>
        </div>
      )}
    </div>
  );
}

function Control({ label, value, min, max, step, onChange }) {
  return (
    <div className="control">
      <div className="control-header">
        <span>{label}</span>
        <span className="value">{value.toFixed(1)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}

function SelectControl({ label, value, options, onChange }) {
  return (
    <label className="select-control">
      <span>{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </label>
  );
}
