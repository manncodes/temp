import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      endpoint,
      model,
      messages,
      temperature = 0.7,
      maxTokens = 1024,
      apiKey,
    } = body;

    if (!endpoint || !model || !messages) {
      return NextResponse.json(
        { error: "Missing required fields: endpoint, model, messages" },
        { status: 400 }
      );
    }

    const client = new OpenAI({
      baseURL: `${endpoint}/v1`,
      apiKey: apiKey || "EMPTY",
    });

    const completion = await client.chat.completions.create({
      model,
      messages,
      temperature,
      max_tokens: maxTokens,
    });

    const content = completion.choices[0]?.message?.content || "";

    return NextResponse.json({
      content,
      usage: completion.usage,
      model: completion.model,
      finishReason: completion.choices[0]?.finish_reason,
    });
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
