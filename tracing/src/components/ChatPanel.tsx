"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Loader2, Settings2 } from "lucide-react";
import { VllmDeployment, ChatMessage } from "@/lib/types";

interface Props {
  deployment: VllmDeployment | null;
  onResponse: (text: string) => void;
}

export default function ChatPanel({ deployment, onResponse }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(1024);
  const [apiKey, setApiKey] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send() {
    if (!input.trim() || !deployment || deployment.status !== "online") return;

    const userMsg: ChatMessage = { role: "user", content: input.trim() };
    const updated = [...messages, userMsg];
    setMessages(updated);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          endpoint: deployment.endpoint,
          model: deployment.model,
          messages: updated,
          temperature,
          maxTokens,
          apiKey: apiKey || undefined,
        }),
      });

      const data = await res.json();
      if (data.error) throw new Error(data.error);

      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: data.content,
      };
      setMessages([...updated, assistantMsg]);
      onResponse(data.content);
    } catch (e: unknown) {
      const errMsg = e instanceof Error ? e.message : "Request failed";
      setMessages([
        ...updated,
        { role: "assistant", content: `Error: ${errMsg}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-card)] flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[var(--border)] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Bot className="w-4 h-4 text-[var(--accent)]" />
          <h2 className="font-semibold text-sm">Chat</h2>
          {deployment && (
            <span className="text-xs text-[var(--text-muted)] ml-2">
              {deployment.model}
            </span>
          )}
        </div>
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="p-1 rounded hover:bg-[var(--bg-card-hover)] transition-colors"
        >
          <Settings2 className="w-4 h-4 text-[var(--text-muted)]" />
        </button>
      </div>

      {/* Settings drawer */}
      {showSettings && (
        <div className="px-4 py-3 border-b border-[var(--border)] space-y-2 bg-[var(--bg)]">
          <div className="flex gap-4">
            <div className="flex-1">
              <label className="block text-xs text-[var(--text-muted)] mb-1">
                Temperature
              </label>
              <input
                type="number"
                min={0}
                max={2}
                step={0.1}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="w-full px-2 py-1 text-sm rounded bg-[var(--bg-card)] border border-[var(--border)] outline-none"
              />
            </div>
            <div className="flex-1">
              <label className="block text-xs text-[var(--text-muted)] mb-1">
                Max Tokens
              </label>
              <input
                type="number"
                min={1}
                max={8192}
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
                className="w-full px-2 py-1 text-sm rounded bg-[var(--bg-card)] border border-[var(--border)] outline-none"
              />
            </div>
          </div>
          <div>
            <label className="block text-xs text-[var(--text-muted)] mb-1">
              API Key (if required)
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Leave blank for vLLM default"
              className="w-full px-2 py-1 text-sm rounded bg-[var(--bg-card)] border border-[var(--border)] outline-none"
            />
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
        {!deployment && (
          <div className="text-center text-[var(--text-muted)] text-sm py-12">
            Select a vLLM deployment from the scanner to start chatting.
          </div>
        )}
        {deployment && messages.length === 0 && (
          <div className="text-center text-[var(--text-muted)] text-sm py-12">
            Connected to <strong>{deployment.model}</strong>. Send a message to
            begin.
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex gap-3 ${msg.role === "user" ? "justify-end" : ""}`}
          >
            {msg.role === "assistant" && (
              <div className="shrink-0 w-7 h-7 rounded-full bg-[var(--accent)]/20 flex items-center justify-center">
                <Bot className="w-4 h-4 text-[var(--accent)]" />
              </div>
            )}
            <div
              className={`max-w-[80%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap ${
                msg.role === "user"
                  ? "bg-[var(--accent)] text-white"
                  : "bg-[var(--bg)] border border-[var(--border)]"
              }`}
            >
              {msg.content}
            </div>
            {msg.role === "user" && (
              <div className="shrink-0 w-7 h-7 rounded-full bg-[var(--bg)] border border-[var(--border)] flex items-center justify-center">
                <User className="w-4 h-4 text-[var(--text-muted)]" />
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="flex gap-3">
            <div className="shrink-0 w-7 h-7 rounded-full bg-[var(--accent)]/20 flex items-center justify-center">
              <Loader2 className="w-4 h-4 text-[var(--accent)] animate-spin" />
            </div>
            <div className="bg-[var(--bg)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-muted)]">
              Generating…
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-3 border-t border-[var(--border)]">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            send();
          }}
          className="flex gap-2"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={!deployment || deployment.status !== "online" || loading}
            placeholder={
              deployment
                ? "Type a message…"
                : "Select a deployment first"
            }
            className="flex-1 px-3 py-2 text-sm rounded bg-[var(--bg)] border border-[var(--border)] focus:border-[var(--accent)] outline-none disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={
              !input.trim() ||
              !deployment ||
              deployment.status !== "online" ||
              loading
            }
            className="px-3 py-2 rounded bg-[var(--accent)] hover:bg-[var(--accent-light)] disabled:opacity-30 transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>
    </div>
  );
}
