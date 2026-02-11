import { NextRequest, NextResponse } from "next/server";
import { scanCluster } from "@/lib/scanner";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const namespaces: string = body.namespaces || "";
    const manualEndpoints: string = body.manualEndpoints || "";
    const defaultPort: number = body.defaultPort || 8000;

    const deployments = await scanCluster(
      namespaces,
      manualEndpoints,
      defaultPort
    );

    return NextResponse.json({ deployments });
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
