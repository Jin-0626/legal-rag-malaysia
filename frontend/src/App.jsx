import { useEffect, useState } from "react";
import { fetchHealth, sendChatStream } from "./api";
import ChatWindow from "./components/ChatWindow";
import ModeSelector from "./components/ModeSelector";
import SourcePanel from "./components/SourcePanel";

const DEMO_QUERIES = [
  "What does Section 2 of the Employment Act 1955 define?",
  "Which section introduces data portability in the PDPA Amendment Act 2024?",
  "Which Article begins Part III on Citizenship in the Federal Constitution?",
  "Apakah kandungan Perkara 8 dalam Perlembagaan Persekutuan?",
  "What is the minimum wage under the Minimum Wages Order 2024?"
];

export default function App() {
  const [mode, setMode] = useState("auto");
  const [topK, setTopK] = useState(5);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState(DEMO_QUERIES[0]);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchHealth().then(setHealth).catch((err) => setError(err.message));
  }, []);

  async function handleSend(event) {
    event.preventDefault();
    const query = input.trim();
    if (!query || loading) {
      return;
    }

    let assistantIndex = -1;
    setLoading(true);
    setError("");
    setMessages((current) => {
      assistantIndex = current.length + 1;
      return [
        ...current,
        { role: "user", text: query },
        {
          role: "assistant",
          text: "",
          modeUsed: "",
          sources: [],
          graphPath: [],
          warnings: [],
          streaming: true
        }
      ];
    });

    try {
      await sendChatStream({
        query,
        mode,
        top_k: topK,
        onEvent: (eventPayload) => {
          setMessages((current) =>
            current.map((message, index) => {
              if (index !== assistantIndex || message.role !== "assistant") {
                return message;
              }

              if (eventPayload.type === "meta") {
                return {
                  ...message,
                  modeUsed: eventPayload.mode_used || "",
                  sources: eventPayload.sources || [],
                  graphPath: eventPayload.graph_path || [],
                  warnings: eventPayload.warnings || []
                };
              }

              if (eventPayload.type === "token") {
                return {
                  ...message,
                  text: `${message.text}${eventPayload.content || ""}`
                };
              }

              if (eventPayload.type === "done") {
                return {
                  ...message,
                  modeUsed: eventPayload.mode_used || message.modeUsed,
                  sources: eventPayload.sources || message.sources,
                  graphPath: eventPayload.graph_path || message.graphPath,
                  warnings: eventPayload.warnings || [],
                  streaming: false
                };
              }

              return message;
            })
          );
        }
      });
      setInput("");
    } catch (err) {
      setError(err.message);
      setMessages((current) =>
        current.map((message, index) =>
          index === assistantIndex && message.role === "assistant"
            ? {
                ...message,
                text: message.text || "Unable to stream a response right now.",
                streaming: false
              }
            : message
        )
      );
    } finally {
      setLoading(false);
    }
  }

  const latestAssistant = [...messages].reverse().find((message) => message.role === "assistant");

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="panel hero">
          <p className="eyebrow">Malaysia Legal RAG</p>
          <h1>Grounded Legal Chat Demo</h1>
          <p className="muted">
            Uses the existing retrieval stack by default and only reaches for graph-assisted mode on structural legal queries.
          </p>
        </div>

        <div className="panel">
          <ModeSelector mode={mode} setMode={setMode} topK={topK} setTopK={setTopK} />
        </div>

        <div className="panel">
          <h2>System Health</h2>
          {health ? (
            <>
              <ul className="healthList">
                <li>Status: {health.status}</li>
                <li>Indexed chunks: {health.indexed_chunks}</li>
                <li>Ollama: {health.ollama_available ? "Connected" : "Unavailable"}</li>
                <li>Model: {health.model}</li>
                <li>Model available: {health.model_available ? "Yes" : "No"}</li>
                <li>Chat ready: {health.chat_ready ? "Yes" : "No"}</li>
              </ul>
              {health.error ? <p className="muted">Ollama detail: {health.error}</p> : null}
            </>
          ) : (
            <p className="muted">Loading health...</p>
          )}
        </div>

        <div className="panel">
          <h2>Demo Queries</h2>
          <div className="chips">
            {DEMO_QUERIES.map((query) => (
              <button key={query} type="button" className="chip" onClick={() => setInput(query)} disabled={loading}>
                {query}
              </button>
            ))}
          </div>
        </div>
      </aside>

      <main className="main">
        <ChatWindow messages={messages} loading={loading} />
        <form className="composer" onSubmit={handleSend}>
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask about a section, article, amendment, or constitutional provision..."
            rows={4}
            disabled={loading}
          />
          <div className="composerBar">
            <span className="hint">Grounded answers only. No legal advice.</span>
            <button type="submit" disabled={loading}>
              {loading ? "Thinking..." : "Ask"}
            </button>
          </div>
        </form>
        {error ? <div className="errorBanner">{error}</div> : null}
      </main>

      <aside className="sidebar">
        <SourcePanel
          title="Top Sources"
          items={latestAssistant?.sources || []}
          emptyText="Retrieved sources will appear here."
        />
        <SourcePanel
          title="Graph Path"
          items={(latestAssistant?.graphPath || []).map((item, index) => ({
            document: `Step ${index + 1}`,
            heading: item,
            unit_type: "",
            unit_id: "",
            preview: "",
            score: 0
          }))}
          emptyText="Graph steps appear for structural queries."
          compact
        />
        {latestAssistant?.warnings?.length ? (
          <div className="panel">
            <h2>Warnings</h2>
            <ul className="healthList">
              {latestAssistant.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </aside>
    </div>
  );
}
