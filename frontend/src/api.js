const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export async function fetchHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error("Failed to load API health.");
  }
  return response.json();
}

export async function sendChat({ query, mode, top_k }) {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ query, mode, top_k })
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "Chat request failed.");
  }
  return response.json();
}

export async function sendChatStream({ query, mode, top_k, onEvent }) {
  const response = await fetch(`${API_BASE_URL}/chat_stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ query, mode, top_k })
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "Streaming chat request failed.");
  }
  if (!response.body) {
    throw new Error("Streaming response body is not available.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      onEvent(JSON.parse(trimmed));
    }
  }

  const tail = buffer.trim();
  if (tail) {
    onEvent(JSON.parse(tail));
  }
}
