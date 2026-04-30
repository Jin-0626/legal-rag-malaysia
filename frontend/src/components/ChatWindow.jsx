import { useEffect, useRef } from "react";

function renderStructuredText(text) {
  const sections = text
    .split(/\n\s*\n/)
    .map((block) => block.trim())
    .filter(Boolean);

  return sections.map((section, sectionIndex) => {
    const lines = section.split("\n").map((line) => line.trim()).filter(Boolean);
    if (lines.length === 0) {
      return null;
    }
    const [firstLine, ...rest] = lines;
    const isHeading = firstLine.endsWith(":");
    const bodyLines = isHeading ? rest : lines;

    return (
      <div key={`section-${sectionIndex}`} className="answerSection">
        {isHeading ? <strong className="answerHeading">{firstLine}</strong> : null}
        {bodyLines.map((line, lineIndex) =>
          line.startsWith("- ") ? (
            <ul key={`line-${lineIndex}`} className="answerList">
              <li>{line.slice(2)}</li>
            </ul>
          ) : (
            <p key={`line-${lineIndex}`} className="answerParagraph">
              {line}
            </p>
          )
        )}
      </div>
    );
  });
}

export default function ChatWindow({ messages, loading }) {
  const containerRef = useRef(null);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) {
      return;
    }
    element.scrollTo({
      top: element.scrollHeight,
      behavior: "smooth"
    });
  }, [messages, loading]);

  return (
    <section ref={containerRef} className="chatWindow">
      {messages.length === 0 ? (
        <div className="message assistant">
          <div className="messageCard">
            Ask about a Malaysian Act, constitutional article, amendment, or hierarchy provision.
          </div>
        </div>
      ) : null}
      {messages.map((message, index) => (
        <div key={`${message.role}-${index}`} className={`message ${message.role}`}>
          <div className="messageCard">
            {message.role === "assistant" ? renderStructuredText(message.text || (message.streaming ? "Thinking..." : "")) : message.text}
            {message.streaming ? <span className="typingCursor">▌</span> : null}
            {message.modeUsed ? <span className="messageMeta">Mode used: {message.modeUsed}</span> : null}
          </div>
        </div>
      ))}
      {loading && !messages.some((message) => message.streaming) ? (
        <div className="message assistant">
          <div className="messageCard">Thinking...</div>
        </div>
      ) : null}
    </section>
  );
}
