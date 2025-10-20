import { useState, useCallback, useRef, useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { LlmAnalysisService, type ApiError } from '~/client';
import { useCustomToast } from './useCustomToast';
import { useSSE, type SSEEvent } from './useSSE';
import { buildLLMAnalysisSSEUrl } from '~/utils/sse';

export interface TimelineEvent {
  phase: string;
  frame: number;
  timestamp: number;
  speed_mph?: number;
  distance_m?: number;
  description: string;
}

export interface WeatherData {
  temperature_f: number;
  condition: string;
  precipitation: string;
  visibility_mi: number;
  road_condition: string;
}

export interface LLMEvent {
  type:
    | 'thinking_start'
    | 'thinking_content'
    | 'thinking_end'
    | 'tool_call_start'
    | 'tool_call_reasoning'
    | 'tool_call_result'
    | 'report_start'
    | 'report_content'
    | 'report_end'
    | 'collision_detected'
    | 'error';
  data: any;
  timestamp: string;
}

export interface AnalysisState {
  isRunning: boolean;
  analysisId: string | null;
  currentPhase:
    | 'idle'
    | 'thinking'
    | 'tool_calls'
    | 'reporting'
    | 'complete'
    | 'error';
  thinkingContent: string;
  toolCalls: Array<{
    id: string;
    tool: string;
    input: any;
    result?: any;
    reasoning?: string;
    status: 'pending' | 'running' | 'completed' | 'error';
  }>;
  reportContent: string;
  collisionResult: string | null;
  timeline: TimelineEvent[] | null;
  weatherData: WeatherData | null;
  error: string | null;
  showThinking: boolean;
  showToolCalls: boolean;
}

export const useLLMAnalysis = (projectId: string, runId?: string) => {
  const { showToast } = useCustomToast();
  const { connect: connectSSE, disconnect: disconnectSSE } = useSSE();

  const [state, setState] = useState<AnalysisState>({
    isRunning: false,
    analysisId: null,
    currentPhase: 'idle',
    thinkingContent: '',
    toolCalls: [],
    reportContent: '',
    collisionResult: null,
    timeline: null,
    weatherData: null,
    error: null,
    showThinking: true,
    showToolCalls: true,
  });

  // Load stored analysis data on mount
  const { data: analysesData } = useAnalysisList(projectId);

  useEffect(() => {
    // Load the most recent completed analysis for this project
    if (analysesData && analysesData.analyses.length > 0) {
      const latestAnalysis = analysesData.analyses[0]; // Already sorted by created_at desc

      if (latestAnalysis.status === 'completed' && latestAnalysis.result_data) {
        console.log('Loading stored analysis data:', latestAnalysis);

        // Extract frontend data from stored analysis
        const frontendData = latestAnalysis.result_data?.frontend_data;
        if (frontendData) {
          // Map timeline data to ensure proper structure
          const timelineData = (frontendData as any).timeline;
          const mappedTimeline = timelineData
            ? timelineData.map((item: any) => ({
                phase: item.phase || item.stage || 'Unknown',
                frame: item.frame || 0,
                timestamp: item.timestamp || 0,
                speed_mph: item.speed_mph,
                distance_m: item.distance_m,
                description:
                  item.description || item.narrative || 'No description',
              }))
            : null;

          setState((prev) => ({
            ...prev,
            analysisId: latestAnalysis.analysis_id,
            currentPhase: 'complete',
            reportContent: (frontendData as any).analysis_text || '',
            timeline: mappedTimeline,
            weatherData: (frontendData as any).weather_data || null,
            collisionResult: (frontendData as any).collision_detected
              ? 'Collision detected'
              : null,
            // Convert tool calls from execution log
            toolCalls: (frontendData as any).tool_calls
              ? (frontendData as any).tool_calls.map(
                  (logEntry: any, index: number) => ({
                    id: `stored_${index}`,
                    tool: logEntry.tool || 'unknown',
                    input: logEntry.input || {},
                    result: logEntry.result || {},
                    reasoning: logEntry.reasoning || '',
                    status: 'completed' as const,
                  })
                )
              : [],
          }));
        } else {
          // Fallback: try to extract data from legacy format
          const resultData = latestAnalysis.result_data;
          if (resultData) {
            // Map timeline data to ensure proper structure for legacy format too
            const timelineData = (resultData as any).timeline;
            const mappedTimeline = timelineData
              ? timelineData.map((item: any) => ({
                  phase: item.phase || item.stage || 'Unknown',
                  frame: item.frame || 0,
                  timestamp: item.timestamp || 0,
                  speed_mph: item.speed_mph,
                  distance_m: item.distance_m,
                  description:
                    item.description || item.narrative || 'No description',
                }))
              : null;

            setState((prev) => ({
              ...prev,
              analysisId: latestAnalysis.analysis_id,
              currentPhase: 'complete',
              reportContent:
                (resultData as any).report ||
                (resultData as any).analysis ||
                '',
              timeline: mappedTimeline,
              weatherData: (resultData as any).weather_data || null,
              collisionResult: (resultData as any).collision_detected
                ? 'Collision detected'
                : null,
            }));
          }
        }
      }
    }
  }, [analysesData]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnectSSE();
    };
  }, [disconnectSSE]);

  const handleLLMEvent = useCallback((event: LLMEvent) => {
    console.log('LLM Event received:', event);
    setState((prev) => {
      switch (event.type) {
        case 'thinking_start':
          return {
            ...prev,
            currentPhase: 'thinking',
            thinkingContent: '',
          };

        case 'thinking_content':
          return {
            ...prev,
            thinkingContent: prev.thinkingContent + event.data.content,
          };

        case 'thinking_end':
          return {
            ...prev,
            currentPhase: 'tool_calls',
          };

        case 'tool_call_start':
          return {
            ...prev,
            currentPhase: 'tool_calls',
            toolCalls: [
              ...prev.toolCalls,
              {
                id: event.data.tool + '_' + Date.now(),
                tool: event.data.tool,
                input: event.data.input,
                status: 'running',
              },
            ],
          };

        case 'tool_call_reasoning':
          return {
            ...prev,
            toolCalls: prev.toolCalls.map((tc) =>
              tc.tool === event.data.tool && tc.status === 'running'
                ? { ...tc, reasoning: event.data.reasoning }
                : tc
            ),
          };

        case 'tool_call_result':
          const updatedToolCalls = prev.toolCalls.map((tc) =>
            tc.tool === event.data.tool && tc.status === 'running'
              ? {
                  ...tc,
                  result: event.data.result,
                  status: 'completed' as const,
                }
              : tc
          );

          // Extract timeline data from build_timeline tool result
          let newTimeline = prev.timeline;
          if (
            event.data.tool === 'build_timeline' &&
            event.data.result?.success &&
            event.data.result?.timeline
          ) {
            newTimeline = event.data.result.timeline.map((item: any) => ({
              phase: item.phase || item.stage || 'Unknown',
              frame: item.frame || 0,
              timestamp: item.timestamp || 0,
              speed_mph: item.speed_mph,
              distance_m: item.distance_m,
              description:
                item.description || item.narrative || 'No description',
            }));
          }

          // Extract weather data from get_weather_data tool result
          let newWeatherData = prev.weatherData;
          if (
            event.data.tool === 'get_weather_data' &&
            event.data.result?.success &&
            event.data.result?.weather_data
          ) {
            const weather = event.data.result.weather_data;
            newWeatherData = {
              temperature_f: weather.temperature_f,
              condition: weather.condition,
              precipitation: weather.precipitation,
              visibility_mi: weather.visibility_mi,
              road_condition: weather.road_condition,
            };
          }

          return {
            ...prev,
            toolCalls: updatedToolCalls,
            timeline: newTimeline,
            weatherData: newWeatherData,
          };

        case 'report_start':
          return {
            ...prev,
            currentPhase: 'reporting',
            reportContent: '',
          };

        case 'report_content':
          return {
            ...prev,
            reportContent: prev.reportContent + event.data.content,
          };

        case 'report_end':
          return {
            ...prev,
            currentPhase: 'complete',
            isRunning: false,
          };

        case 'collision_detected':
          return {
            ...prev,
            collisionResult: event.data.message || 'Collision detection result',
          };

        case 'error':
          return {
            ...prev,
            error: event.data.message || 'An error occurred during analysis',
            isRunning: false,
            currentPhase: 'error',
          };

        default:
          return prev;
      }
    });
  }, []);

  // Mutation for starting LLM analysis
  const startAnalysisMutation = useMutation({
    mutationFn: async () => {
      if (!runId) {
        throw new Error('No processing run available for analysis');
      }
      return LlmAnalysisService.startLlmAnalysis({
        projectId,
        requestBody: { run_id: runId },
      });
    },
    onError: (error: ApiError) => {
      console.error('Failed to start analysis:', error);
      setState((prev) => ({
        ...prev,
        error: error.message || 'Failed to start analysis',
        isRunning: false,
      }));
      showToast(error.message || 'Failed to start analysis', 'error');
    },
  });

  const startAnalysis = useCallback(async () => {
    if (!runId) {
      showToast('No processing run available for analysis', 'error');
      return;
    }

    try {
      setState((prev) => ({
        ...prev,
        isRunning: true,
        currentPhase: 'thinking',
        thinkingContent: '',
        toolCalls: [],
        reportContent: '',
        collisionResult: null,
        error: null,
      }));

      // Start the analysis using the mutation
      const response = await startAnalysisMutation.mutateAsync();
      const analysisId = response.analysis_id;

      setState((prev) => ({ ...prev, analysisId }));

      // Connect to SSE stream using the generated client URL pattern
      const sseUrl = buildLLMAnalysisSSEUrl(projectId, analysisId);

      connectSSE(sseUrl, {
        onMessage: (sseEvent: SSEEvent) => {
          const llmEvent: LLMEvent = {
            type: sseEvent.type as any,
            data: sseEvent.data,
            timestamp: sseEvent.timestamp || new Date().toISOString(),
          };
          handleLLMEvent(llmEvent);
        },
        onError: (error) => {
          console.error('SSE connection error:', error);
          setState((prev) => ({
            ...prev,
            error: 'Connection lost. Please try again.',
            isRunning: false,
          }));
        },
        onOpen: () => {
          console.log('SSE connection opened for LLM analysis');
        },
        onClose: () => {
          console.log('SSE connection closed gracefully');
        },
      });
    } catch (error) {
      console.error('Failed to start analysis:', error);
      setState((prev) => ({
        ...prev,
        error:
          error instanceof Error ? error.message : 'Failed to start analysis',
        isRunning: false,
      }));
    }
  }, [projectId, runId, showToast, handleLLMEvent, startAnalysisMutation]);

  const stopAnalysis = useCallback(() => {
    disconnectSSE();
    setState((prev) => ({
      ...prev,
      isRunning: false,
      currentPhase: 'idle',
    }));
  }, [disconnectSSE]);

  const resetAnalysis = useCallback(() => {
    disconnectSSE();
    setState({
      isRunning: false,
      analysisId: null,
      currentPhase: 'idle',
      thinkingContent: '',
      toolCalls: [],
      reportContent: '',
      collisionResult: null,
      timeline: null,
      weatherData: null,
      error: null,
      showThinking: true,
      showToolCalls: true,
    });
  }, [disconnectSSE]);

  const toggleThinking = useCallback(() => {
    setState((prev) => ({ ...prev, showThinking: !prev.showThinking }));
  }, []);

  const toggleToolCalls = useCallback(() => {
    setState((prev) => ({ ...prev, showToolCalls: !prev.showToolCalls }));
  }, []);

  return {
    state,
    startAnalysis,
    stopAnalysis,
    resetAnalysis,
    toggleThinking,
    toggleToolCalls,
    isStarting: startAnalysisMutation.isPending,
  };
};

/**
 * Hook to list stored LLM analyses for a project.
 */
export function useAnalysisList(projectId: string, skip = 0, limit = 100) {
  return useQuery({
    queryKey: ['llm-analyses', projectId, skip, limit],
    queryFn: async () => {
      const response = await LlmAnalysisService.listAnalyses({
        projectId,
        skip,
        limit,
      });
      return response;
    },
    enabled: !!projectId,
  });
}

/**
 * Hook to get a specific LLM analysis result.
 */
export function useAnalysisResult(projectId: string, analysisId: string) {
  return useQuery({
    queryKey: ['llm-analysis', projectId, analysisId],
    queryFn: async () => {
      const response = await LlmAnalysisService.getAnalysis({
        projectId,
        analysisId,
      });
      return response;
    },
    enabled: !!projectId && !!analysisId,
  });
}

/**
 * Hook to delete/reset an LLM analysis.
 */
export function useDeleteAnalysis(projectId: string) {
  const queryClient = useQueryClient();
  const { showToast } = useCustomToast();

  return useMutation({
    mutationFn: async (analysisId: string) => {
      const response = await LlmAnalysisService.deleteAnalysis({
        projectId,
        analysisId,
      });
      return response;
    },
    onSuccess: () => {
      showToast('Analysis deleted successfully', 'success');
      // Invalidate analyses list to refresh
      queryClient.invalidateQueries({ queryKey: ['llm-analyses', projectId] });
    },
    onError: (error) => {
      showToast(`Failed to delete analysis: ${error.message}`, 'error');
    },
  });
}
