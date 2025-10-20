import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ReportsService } from '~/client/sdk.gen';
import {
  type ReportResponse,
  type ReportListResponse,
  type GenerateReportRequest,
  type GenerateReportResponse,
} from '~/client/types.gen';
import { useCustomToast } from './useCustomToast';

// Use types from generated client

/**
 * Hook to generate a PDF report from LLM analysis results.
 */
export function useGenerateReport(projectId: string) {
  const queryClient = useQueryClient();
  const { showToast } = useCustomToast();

  return useMutation({
    mutationFn: async (
      request: GenerateReportRequest
    ): Promise<GenerateReportResponse> => {
      const response = await ReportsService.generateReport({
        projectId,
        requestBody: request,
      });

      return response;
    },
    onSuccess: (data) => {
      showToast('PDF report generation started successfully', 'success');
      // Invalidate reports list to refresh
      queryClient.invalidateQueries({ queryKey: ['reports', projectId] });
    },
    onError: (error) => {
      showToast(`Failed to generate report: ${error.message}`, 'error');
    },
  });
}

/**
 * Hook to list all reports for a project.
 */
export function useReports(projectId: string, skip = 0, limit = 100) {
  return useQuery({
    queryKey: ['reports', projectId, skip, limit],
    queryFn: async (): Promise<ReportListResponse> => {
      const response = await ReportsService.listReports({
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
 * Hook to get specific report details.
 */
export function useReport(projectId: string, reportId: string) {
  return useQuery({
    queryKey: ['report', projectId, reportId],
    queryFn: async (): Promise<ReportResponse> => {
      const response = await ReportsService.getReportDetails({
        projectId,
        reportId,
      });

      return response;
    },
    enabled: !!projectId && !!reportId,
  });
}

/**
 * Hook to download a report PDF.
 */
export function useDownloadReport(projectId: string, reportId: string) {
  const { showToast } = useCustomToast();

  return useMutation({
    mutationFn: async (): Promise<unknown> => {
      const response = await ReportsService.downloadReport({
        projectId,
        reportId,
      });

      // The response should be a redirect, but we'll handle it in the component
      return response;
    },
    onError: (error) => {
      showToast(`Failed to download report: ${error.message}`, 'error');
    },
  });
}

/**
 * Utility function to get report status badge color.
 */
export function getReportStatusColor(status: string): string {
  switch (status) {
    case 'pending':
      return 'yellow';
    case 'generating':
      return 'blue';
    case 'completed':
      return 'green';
    case 'failed':
      return 'red';
    default:
      return 'gray';
  }
}

/**
 * Utility function to format report status for display.
 */
export function formatReportStatus(status: string): string {
  switch (status) {
    case 'pending':
      return 'Pending';
    case 'generating':
      return 'Generating';
    case 'completed':
      return 'Completed';
    case 'failed':
      return 'Failed';
    default:
      return 'Unknown';
  }
}
