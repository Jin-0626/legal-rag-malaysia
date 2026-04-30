export default function SourcePanel({ title, items, emptyText, compact = false }) {
  return (
    <section className="panel">
      <h2>{title}</h2>
      {items.length === 0 ? (
        <p className="muted">{emptyText}</p>
      ) : (
        <div className="sourceList">
          {items.map((item, index) => (
            <article className="sourceCard" key={`${title}-${index}-${item.heading}`}>
              <div className="sourceTop">
                <strong>{item.document}</strong>
                {!compact ? <span className="sourceMeta">score {item.score}</span> : null}
              </div>
              {item.unit_type || item.unit_id ? (
                <div className="sourceMeta">
                  {[item.unit_type, item.unit_id].filter(Boolean).join(" ")}
                  {item.chunk_count > 1 ? ` (${item.chunk_count} chunks)` : ""}
                </div>
              ) : null}
              <p className="sourceHeading">{item.heading}</p>
              {item.preview ? <p className="sourcePreview">{item.preview}</p> : null}
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
