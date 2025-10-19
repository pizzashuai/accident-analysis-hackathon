import { OpenAPI } from '~/client';

/**
 * Constructs the SSE URL for LLM analysis events using the same base URL
 * as the generated client to ensure consistency.
 */
export function buildLLMAnalysisSSEUrl(
  projectId: string,
  analysisId: string
): string {
  // Use the same base URL as the OpenAPI client, with fallback
  const baseUrl =
    OpenAPI.BASE || import.meta.env.VITE_API_URL || 'http://localhost:8000';
  return `${baseUrl}/api/v1/llm-analysis/projects/${projectId}/stream/${analysisId}`;
}
