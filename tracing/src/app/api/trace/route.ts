import { NextRequest, NextResponse } from "next/server";
import { traceResponse, countNgram, searchDocs } from "@/lib/infinigram";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { text, index, mode = "full" } = body;

    if (!text) {
      return NextResponse.json(
        { error: "Missing required field: text" },
        { status: 400 }
      );
    }

    if (mode === "count") {
      const result = await countNgram(text, index);
      return NextResponse.json(result);
    }

    if (mode === "search") {
      const result = await searchDocs(text, 5, index);
      return NextResponse.json(result);
    }

    // Full trace mode: split response into chunks, score each, find sources
    const result = await traceResponse(text, index);
    return NextResponse.json(result);
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
