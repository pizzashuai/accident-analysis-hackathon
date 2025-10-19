import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  ProjectsService,
  type HomographySessionPublic,
  type HomographyPairPublic,
  type HomographyPairCreate,
  type HomographyModelPublic,
  type HomographySolveResponse,
  type MediaAssetPublic,
  type ApiError,
} from '~/client';
import { handleError } from '~/utils';

export function useHomographySession(projectId: string) {
  return useQuery<HomographySessionPublic | null, ApiError>({
    queryKey: ['homography-session', projectId],
    queryFn: async () => {
      try {
        return await ProjectsService.getHomographySession({ projectId });
      } catch (error: any) {
        // Handle 404 errors gracefully - no session exists yet
        if (error?.status === 404) {
          return null;
        }
        throw error;
      }
    },
    enabled: !!projectId,
  });
}

export function useCreateHomographySession() {
  const queryClient = useQueryClient();

  return useMutation<HomographySessionPublic, ApiError, string>({
    mutationFn: (projectId) =>
      ProjectsService.createHomographySession({ projectId }),
    onSuccess: (_, projectId) => {
      queryClient.invalidateQueries({
        queryKey: ['homography-session', projectId],
      });
      queryClient.invalidateQueries({ queryKey: ['project', projectId] });
    },
    onError: (err: ApiError) => {
      handleError(err);
    },
  });
}

export function useAddHomographyPair() {
  const queryClient = useQueryClient();

  return useMutation<
    HomographyPairPublic,
    ApiError,
    { sessionId: string; pairData: HomographyPairCreate }
  >({
    mutationFn: ({ sessionId, pairData }) =>
      ProjectsService.addHomographyPair({
        sessionId,
        requestBody: pairData,
      }),
    onSuccess: (_, { sessionId }) => {
      queryClient.invalidateQueries({ queryKey: ['homography-session'] });
      queryClient.invalidateQueries({
        queryKey: ['homography-pairs', sessionId],
      });
    },
    onError: (err: ApiError) => {
      handleError(err);
    },
  });
}

export function useUpdateHomographyPairs() {
  const queryClient = useQueryClient();

  return useMutation<
    HomographyPairPublic[],
    ApiError,
    { sessionId: string; pairsData: HomographyPairCreate[] }
  >({
    mutationFn: ({ sessionId, pairsData }) =>
      ProjectsService.updateHomographyPairs({
        sessionId,
        requestBody: pairsData,
      }),
    onSuccess: (_, { sessionId }) => {
      queryClient.invalidateQueries({ queryKey: ['homography-session'] });
      queryClient.invalidateQueries({
        queryKey: ['homography-pairs', sessionId],
      });
    },
    onError: (err: ApiError) => {
      handleError(err);
    },
  });
}

export function useDeleteHomographyPair() {
  const queryClient = useQueryClient();

  return useMutation<{ message: string }, ApiError, string>({
    mutationFn: async (pairId) => {
      const result = await ProjectsService.deleteHomographyPair({ pairId });
      return { message: 'Pair deleted successfully' };
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['homography-session'] });
      queryClient.invalidateQueries({ queryKey: ['homography-pairs'] });
    },
    onError: (err: ApiError) => {
      handleError(err);
    },
  });
}

export function useSolveHomography() {
  const queryClient = useQueryClient();

  return useMutation<HomographySolveResponse, ApiError, string>({
    mutationFn: (sessionId) =>
      ProjectsService.solveHomographySession({ sessionId }),
    onSuccess: (_, sessionId) => {
      queryClient.invalidateQueries({ queryKey: ['homography-session'] });
      queryClient.invalidateQueries({
        queryKey: ['homography-model', sessionId],
      });
    },
    onError: (err: ApiError) => {
      handleError(err);
    },
  });
}

export function useHomographyModel(sessionId: string) {
  return useQuery<HomographyModelPublic, ApiError>({
    queryKey: ['homography-model', sessionId],
    queryFn: () => ProjectsService.getHomographyModel({ sessionId }),
    enabled: !!sessionId,
  });
}

export function useExtractFrame() {
  const queryClient = useQueryClient();

  return useMutation<MediaAssetPublic, ApiError, string>({
    mutationFn: (projectId) => ProjectsService.extractVideoFrame({ projectId }),
    onSuccess: (_, projectId) => {
      queryClient.invalidateQueries({ queryKey: ['project', projectId] });
      queryClient.invalidateQueries({
        queryKey: ['homography-session', projectId],
      });
    },
    onError: (err: ApiError) => {
      handleError(err);
    },
  });
}

export function useExportHomography() {
  return useMutation<unknown, ApiError, string>({
    mutationFn: (sessionId) =>
      ProjectsService.exportHomographySession({ sessionId }),
    onError: (err: ApiError) => {
      handleError(err);
    },
  });
}
