import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  ProjectsService,
  type ProjectPublic,
  type ProjectsPublic,
  type MediaAssetPublic,
  type ProjectLocationPublic,
  type ApiError,
  type src__common__features__project__schemas__Message,
} from '~/client';
import { handleError } from '~/utils';

export function useProjects(skip = 0, limit = 100) {
  return useQuery<ProjectsPublic, ApiError>({
    queryKey: ['projects', skip, limit],
    queryFn: () => ProjectsService.readProjects({ skip, limit }),
  });
}

export function useProject(projectId: string) {
  return useQuery<ProjectPublic, ApiError>({
    queryKey: ['project', projectId],
    queryFn: () => ProjectsService.readProject({ projectId }),
    enabled: !!projectId,
  });
}

export function useCreateProject() {
  const queryClient = useQueryClient();

  return useMutation<
    ProjectPublic,
    ApiError,
    { title: string; description?: string }
  >({
    mutationFn: (data) =>
      ProjectsService.createProjectRoute({ requestBody: data }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
    },
    onError: (err: ApiError) => {
      handleError(err);
    },
  });
}

export function useUpdateProject() {
  const queryClient = useQueryClient();

  return useMutation<
    ProjectPublic,
    ApiError,
    { projectId: string; data: { title?: string; description?: string } }
  >({
    mutationFn: ({ projectId, data }) =>
      ProjectsService.updateProjectRoute({ projectId, requestBody: data }),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
      queryClient.invalidateQueries({ queryKey: ['project', projectId] });
    },
    onError: (err: ApiError) => {
      handleError(err);
    },
  });
}

export function useDeleteProject() {
  const queryClient = useQueryClient();

  return useMutation<
    src__common__features__project__schemas__Message,
    ApiError,
    string
  >({
    mutationFn: (projectId) =>
      ProjectsService.deleteProjectRoute({ projectId }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
    },
    onError: (err: ApiError) => {
      handleError(err);
    },
  });
}

export function useUploadVideo() {
  const queryClient = useQueryClient();

  return useMutation<
    MediaAssetPublic,
    ApiError,
    { projectId: string; file: File }
  >({
    mutationFn: ({ projectId, file }) => {
      return ProjectsService.uploadVideo({ projectId, formData: { file } });
    },
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: ['project', projectId] });
      queryClient.invalidateQueries({ queryKey: ['projects'] });
    },
    onError: (err: ApiError) => {
      handleError(err);
    },
  });
}

export function useSetProjectLocation() {
  const queryClient = useQueryClient();

  return useMutation<
    ProjectLocationPublic,
    ApiError,
    {
      projectId: string;
      locationData: {
        addr_line?: string;
        lat?: number;
        lon?: number;
        source?: string;
      };
    }
  >({
    mutationFn: ({ projectId, locationData }) =>
      ProjectsService.setProjectLocation({
        projectId,
        requestBody: locationData,
      }),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: ['project', projectId] });
      queryClient.invalidateQueries({ queryKey: ['projects'] });
    },
    onError: (err: ApiError) => {
      handleError(err);
    },
  });
}

export function useMediaPresignedUrl() {
  return useMutation<
    unknown,
    ApiError,
    {
      projectId: string;
      mediaAssetId: string;
    }
  >({
    mutationFn: ({ projectId, mediaAssetId }) =>
      ProjectsService.getMediaPresignedUrl({
        projectId,
        mediaAssetId,
      }),
    onError: (err: ApiError) => {
      handleError(err);
    },
  });
}
