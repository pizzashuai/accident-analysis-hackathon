import { useState } from 'react';
import {
  Card,
  Title,
  Text,
  Button,
  Group,
  Badge,
  Table,
  Stack,
  Alert,
  LoadingOverlay,
  ActionIcon,
  Tooltip,
  Modal,
  Box,
} from '@mantine/core';
import {
  IconFileText,
  IconDownload,
  IconRefresh,
  IconAlertCircle,
  IconCheck,
  IconClock,
  IconX,
} from '@tabler/icons-react';
import {
  useGenerateReport,
  useReports,
  useDownloadReport,
  getReportStatusColor,
  formatReportStatus,
} from '~/hooks/useReports';
import { type ReportResponse } from '~/client/types.gen';
import { useCustomToast } from '~/hooks/useCustomToast';

interface ReportGenerationPanelProps {
  projectId: string;
  analysisId?: string;
  runId?: string;
  disabled?: boolean;
}

export function ReportGenerationPanel({
  projectId,
  analysisId,
  runId,
  disabled = false,
}: ReportGenerationPanelProps) {
  const [downloadModalOpened, setDownloadModalOpened] = useState(false);
  const [selectedReport, setSelectedReport] = useState<ReportResponse | null>(
    null
  );

  const generateReport = useGenerateReport(projectId);
  const {
    data: reportsData,
    isLoading: isLoadingReports,
    refetch,
  } = useReports(projectId);
  const downloadReport = useDownloadReport(projectId, selectedReport?.id || '');
  const { showToast } = useCustomToast();

  const handleGenerateReport = async () => {
    if (!analysisId || !runId) {
      showToast(
        'Analysis ID and Run ID are required to generate a report',
        'error'
      );
      return;
    }

    try {
      await generateReport.mutateAsync({
        analysis_id: analysisId,
        run_id: runId,
      });
    } catch (error) {
      // Error handling is done in the hook
    }
  };

  const handleDownloadReport = async (report: ReportResponse) => {
    if (!report.pdf_uri) {
      showToast('PDF not available for this report', 'error');
      return;
    }

    // Open PDF in new tab
    window.open(report.pdf_uri, '_blank');
  };

  const handleRefreshReports = () => {
    refetch();
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending':
        return <IconClock size={16} />;
      case 'generating':
        return <IconRefresh size={16} className='animate-spin' />;
      case 'completed':
        return <IconCheck size={16} />;
      case 'failed':
        return <IconX size={16} />;
      default:
        return <IconAlertCircle size={16} />;
    }
  };

  const canGenerateReport =
    !disabled && analysisId && runId && !generateReport.isPending;
  const hasReports = reportsData && reportsData.reports.length > 0;

  return (
    <Card withBorder>
      <Stack gap='md'>
        <Group justify='space-between' align='center'>
          <Group gap='sm'>
            <IconFileText size={20} />
            <Title order={4}>PDF Reports</Title>
          </Group>
          <Group gap='sm'>
            <Tooltip label='Refresh reports'>
              <ActionIcon
                variant='subtle'
                onClick={handleRefreshReports}
                loading={isLoadingReports}
              >
                <IconRefresh size={16} />
              </ActionIcon>
            </Tooltip>
          </Group>
        </Group>

        <Text size='sm' c='dimmed'>
          Generate professional PDF reports with analysis results, screenshots,
          and map overlays. Reports are automatically generated after LLM
          analysis completes.
        </Text>

        {/* Auto-generation info */}
        <Alert icon={<IconFileText size={16} />} color='blue' variant='light'>
          <Text size='sm'>
            <strong>Auto-Generation:</strong> PDF reports are automatically
            created when LLM analysis completes. You can also manually generate
            additional reports using the button below.
          </Text>
        </Alert>

        {!analysisId || !runId ? (
          <Alert icon={<IconAlertCircle size={16} />} color='yellow'>
            Complete LLM analysis first to generate PDF reports.
          </Alert>
        ) : (
          <Group>
            <Button
              leftSection={<IconFileText size={16} />}
              onClick={handleGenerateReport}
              loading={generateReport.isPending}
              disabled={!canGenerateReport}
            >
              Generate PDF Report
            </Button>
          </Group>
        )}

        {generateReport.isPending && (
          <Alert
            icon={<IconRefresh size={16} className='animate-spin' />}
            color='blue'
          >
            Generating PDF report... This may take a few minutes.
          </Alert>
        )}

        {hasReports && (
          <Box>
            <Title order={5} mb='sm'>
              Generated Reports ({reportsData.reports.length})
            </Title>

            <Table striped highlightOnHover>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Status</Table.Th>
                  <Table.Th>Created</Table.Th>
                  <Table.Th>Analysis ID</Table.Th>
                  <Table.Th>Type</Table.Th>
                  <Table.Th>Actions</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {reportsData.reports.map((report) => (
                  <Table.Tr key={report.id}>
                    <Table.Td>
                      <Badge
                        color={getReportStatusColor(report.status)}
                        leftSection={getStatusIcon(report.status)}
                        variant='light'
                      >
                        {formatReportStatus(report.status)}
                      </Badge>
                    </Table.Td>
                    <Table.Td>
                      <Text size='sm'>{formatDate(report.created_at)}</Text>
                    </Table.Td>
                    <Table.Td>
                      <Text size='sm' c='dimmed' ff='monospace'>
                        {report.analysis_id.slice(0, 8)}...
                      </Text>
                    </Table.Td>
                    <Table.Td>
                      <Badge
                        color={report.meta?.auto_generated ? 'blue' : 'gray'}
                        variant='light'
                        size='sm'
                      >
                        {report.meta?.auto_generated ? 'Auto' : 'Manual'}
                      </Badge>
                    </Table.Td>
                    <Table.Td>
                      <Group gap='xs'>
                        {report.status === 'completed' && report.pdf_uri && (
                          <Tooltip label='Download PDF'>
                            <ActionIcon
                              variant='subtle'
                              color='blue'
                              onClick={() => handleDownloadReport(report)}
                            >
                              <IconDownload size={16} />
                            </ActionIcon>
                          </Tooltip>
                        )}
                        {report.status === 'failed' && (
                          <Tooltip label='Report generation failed'>
                            <ActionIcon variant='subtle' color='red'>
                              <IconAlertCircle size={16} />
                            </ActionIcon>
                          </Tooltip>
                        )}
                      </Group>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Box>
        )}

        {!hasReports && !isLoadingReports && (
          <Alert icon={<IconFileText size={16} />} color='gray'>
            No reports generated yet. Complete LLM analysis and generate your
            first report.
          </Alert>
        )}
      </Stack>

      {/* Download Modal */}
      <Modal
        opened={downloadModalOpened}
        onClose={() => setDownloadModalOpened(false)}
        title='Download Report'
        size='sm'
      >
        {selectedReport && (
          <Stack gap='md'>
            <Text>
              Download the PDF report for analysis ID:{' '}
              {selectedReport.analysis_id}
            </Text>
            <Group justify='flex-end'>
              <Button
                variant='outline'
                onClick={() => setDownloadModalOpened(false)}
              >
                Cancel
              </Button>
              <Button
                leftSection={<IconDownload size={16} />}
                onClick={() => {
                  handleDownloadReport(selectedReport);
                  setDownloadModalOpened(false);
                }}
                loading={downloadReport.isPending}
              >
                Download PDF
              </Button>
            </Group>
          </Stack>
        )}
      </Modal>
    </Card>
  );
}
