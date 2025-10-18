import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { ApiError } from '../client/core/ApiError';
import type {
  ProcessingRunCreate,
  ProcessingRunPublic,
  DetectionsPublic,
  ArtifactsPublic,
} from '../client/types.gen';

// Hook to get processing run with automatic polling for active runs
export function useProcessingRunWithPolling(runId: string) {
  return useQuery({
    queryKey: ['processing-run', runId],
    queryFn: async () => {
      const { ProjectsService } = await import('../client/sdk.gen');
      return ProjectsService.getProcessingRunRoute({
        runId,
      });
    },
    enabled: !!runId,
    refetchInterval: (query) => {
      // Refetch every 2 seconds if the run is still pending or running
      const data = query.state.data;
      if (data && (data.status === 'pending' || data.status === 'running')) {
        return 2000;
      }
      return false;
    },
  });
}

// Hook to start video processing
export function useStartProcessing(projectId: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (params: ProcessingRunCreate) => {
      const { ProjectsService } = await import('../client/sdk.gen');
      return ProjectsService.startProcessing({
        projectId,
        requestBody: params,
      });
    },
    onSuccess: () => {
      // Invalidate processing runs query to refetch the list
      queryClient.invalidateQueries({
        queryKey: ['processing-runs', projectId],
      });
    },
  });
}

// Hook to list processing runs for a project
export function useProcessingRuns(projectId: string, enabled: boolean = true) {
  return useQuery({
    queryKey: ['processing-runs', projectId],
    queryFn: async () => {
      const { ProjectsService } = await import('../client/sdk.gen');
      return ProjectsService.listProcessingRunsRoute({
        projectId,
      });
    },
    enabled: !!projectId && enabled,
  });
}

// Hook to get a single processing run (with polling for active runs)
export function useProcessingRun(runId: string) {
  return useProcessingRunWithPolling(runId);
}

// Hook to get detections for a run
export function useDetections(
  runId: string,
  frameIdx?: number,
  skip: number = 0,
  limit: number = 100
) {
  return useQuery({
    queryKey: ['detections', runId, frameIdx, skip, limit],
    queryFn: async () => {
      const { ProjectsService } = await import('../client/sdk.gen');

      if (frameIdx !== undefined) {
        return ProjectsService.getDetectionsByFrameRoute({
          runId,
          frameIdx,
        });
      } else {
        return ProjectsService.getDetectionsRoute({
          runId,
          skip,
          limit,
          frameIdx,
        });
      }
    },
    enabled: !!runId,
  });
}

// Hook to generate annotated video
export function useGenerateAnnotatedVideo(runId: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      const { ProjectsService } = await import('../client/sdk.gen');
      return ProjectsService.generateAnnotatedVideoRoute({
        runId,
      });
    },
    onSuccess: () => {
      // Invalidate artifacts query to refetch the list
      queryClient.invalidateQueries({ queryKey: ['artifacts', runId] });
    },
  });
}

// Hook to list artifacts for a run
export function useArtifacts(runId: string, autoRefresh: boolean = false) {
  return useQuery({
    queryKey: ['artifacts', runId],
    queryFn: async () => {
      const { ProjectsService } = await import('../client/sdk.gen');
      return ProjectsService.listArtifactsRoute({
        runId,
      });
    },
    enabled: !!runId,
    refetchInterval: autoRefresh ? 3000 : false, // Refresh every 3 seconds if auto-refresh is enabled
  });
}

// Hook to get artifact download URL
export function useArtifactDownloadUrl(
  artifactId: string,
  enabled: boolean = true
) {
  return useQuery({
    queryKey: ['artifact-download', artifactId],
    queryFn: async () => {
      const { ProjectsService } = await import('../client/sdk.gen');
      return ProjectsService.getArtifactDownloadUrl({
        artifactId,
      });
    },
    enabled: !!artifactId && enabled,
  });
}

// Hook to get artifact content directly (proxied from backend to avoid CORS)
export function useArtifactContent(
  artifactId: string,
  enabled: boolean = true
) {
  return useQuery({
    queryKey: ['artifact-content', artifactId],
    queryFn: async () => {
      const { ProjectsService } = await import('../client/sdk.gen');
      return ProjectsService.getArtifactContent({
        artifactId,
      });
    },
    enabled: !!artifactId && enabled,
  });
}

// Hook to download an artifact
export function useDownloadArtifact() {
  return useMutation({
    mutationFn: async (artifactId: string) => {
      const { ProjectsService } = await import('../client/sdk.gen');
      const response = await ProjectsService.getArtifactDownloadUrl({
        artifactId,
      });

      // Create a temporary link to trigger download
      const link = document.createElement('a');
      link.href = (response as { url: string }).url;
      link.download = ''; // Let the server determine the filename
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      return response;
    },
  });
}

// Hook to get JSONL artifacts from all processing runs for a project
export function useProjectJsonlArtifacts(
  projectId: string,
  enabled: boolean = true
) {
  return useQuery({
    queryKey: ['project-jsonl-artifacts', projectId],
    queryFn: async () => {
      const { ProjectsService } = await import('../client/sdk.gen');

      // Get all processing runs for the project
      const runsResponse = await ProjectsService.listProcessingRunsRoute({
        projectId,
      });

      if (!runsResponse.data || runsResponse.data.length === 0) {
        return [];
      }

      // Get artifacts for each completed run
      const artifactPromises = runsResponse.data
        .filter((run: any) => run.status === 'completed')
        .map(async (run: any) => {
          try {
            const artifactsResponse = await ProjectsService.listArtifactsRoute({
              runId: run.id,
            });

            // Filter for JSONL detection artifacts
            const jsonlArtifacts =
              artifactsResponse.data?.filter(
                (artifact: any) => artifact.kind === 'jsonl_detections'
              ) || [];

            return jsonlArtifacts.map((artifact: any) => ({
              ...artifact,
              runId: run.id,
              runStartedAt: run.started_at,
            }));
          } catch (error) {
            console.warn(`Failed to fetch artifacts for run ${run.id}:`, error);
            return [];
          }
        });

      const artifactArrays = await Promise.all(artifactPromises);
      return artifactArrays.flat();
    },
    enabled: !!projectId && enabled,
  });
}
