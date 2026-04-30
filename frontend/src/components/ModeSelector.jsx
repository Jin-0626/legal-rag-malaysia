const OPTIONS = [
  { value: "auto", label: "Auto" },
  { value: "hybrid", label: "Hybrid" },
  { value: "graph", label: "Graph" }
];

export default function ModeSelector({ mode, setMode, topK, setTopK }) {
  return (
    <div className="modeSelector">
      <div>
        <h2>Query Mode</h2>
        <p className="muted">Auto defaults to hybrid rerank and only uses graph routing for structural legal queries.</p>
      </div>
      <div className="modeRow">
        {OPTIONS.map((option) => (
          <button
            key={option.value}
            type="button"
            className={`modePill ${mode === option.value ? "active" : ""}`}
            onClick={() => setMode(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
      <label className="topKField">
        Top-K
        <input
          type="number"
          min="1"
          max="10"
          value={topK}
          onChange={(event) => setTopK(Number(event.target.value) || 5)}
        />
      </label>
    </div>
  );
}
