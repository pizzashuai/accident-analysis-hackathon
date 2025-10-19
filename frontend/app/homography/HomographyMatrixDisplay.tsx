import { useState } from 'react';
import {
  Card,
  Text,
  Badge,
  Stack,
  Group,
  Collapse,
  Button,
  Table,
  NumberFormatter,
} from '@mantine/core';
import { IconChevronDown, IconChevronRight } from '@tabler/icons-react';

interface HomographyMatrixDisplayProps {
  matrix: number[][];
  error?: number;
  inlierCount?: number;
  totalPairs?: number;
}

export function HomographyMatrixDisplay({
  matrix,
  error,
  inlierCount,
  totalPairs,
}: HomographyMatrixDisplayProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const formatScientific = (value: number) => {
    if (Math.abs(value) < 0.001 || Math.abs(value) > 1000) {
      return value.toExponential(3);
    }
    return value.toFixed(6);
  };

  const getErrorColor = (error?: number) => {
    if (!error) return 'gray';
    if (error < 0.001) return 'green';
    if (error < 0.01) return 'yellow';
    return 'red';
  };

  const getErrorLabel = (error?: number) => {
    if (!error) return 'Unknown';
    if (error < 0.001) return 'Excellent';
    if (error < 0.01) return 'Good';
    if (error < 0.1) return 'Fair';
    return 'Poor';
  };

  return (
    <Card shadow='sm' padding='md' radius='md' withBorder>
      <Stack gap='sm'>
        <Group justify='space-between' align='center'>
          <Group gap='xs'>
            <Text fw={500} size='md'>
              Homography Matrix
            </Text>
            <Badge color={getErrorColor(error)} variant='light' size='sm'>
              {getErrorLabel(error)}
            </Badge>
          </Group>

          <Button
            variant='subtle'
            size='xs'
            leftSection={
              isExpanded ? (
                <IconChevronDown size={14} />
              ) : (
                <IconChevronRight size={14} />
              )
            }
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? 'Hide' : 'Show'} Matrix
          </Button>
        </Group>

        <Group gap='md'>
          {error !== undefined && (
            <Group gap={4}>
              <Text size='sm' c='dimmed'>
                Reprojection Error:
              </Text>
              <Badge color={getErrorColor(error)} variant='outline' size='sm'>
                <NumberFormatter value={error} decimalScale={6} />
              </Badge>
            </Group>
          )}

          {inlierCount !== undefined && totalPairs !== undefined && (
            <Group gap={4}>
              <Text size='sm' c='dimmed'>
                Inliers:
              </Text>
              <Badge
                color={inlierCount === totalPairs ? 'green' : 'yellow'}
                variant='outline'
                size='sm'
              >
                {inlierCount}/{totalPairs}
              </Badge>
            </Group>
          )}
        </Group>

        <Collapse in={isExpanded}>
          <Card withBorder>
            <Stack gap='xs'>
              <Text size='sm' fw={500} c='dimmed'>
                3×3 Transformation Matrix
              </Text>

              <Table striped>
                <Table.Tbody>
                  {matrix.map((row, rowIndex) => (
                    <Table.Tr key={rowIndex}>
                      {row.map((value, colIndex) => (
                        <Table.Td
                          key={colIndex}
                          style={{
                            textAlign: 'center',
                            fontFamily: 'monospace',
                          }}
                        >
                          {formatScientific(value)}
                        </Table.Td>
                      ))}
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>

              <Text size='xs' c='dimmed' ta='center'>
                This matrix transforms normalized image coordinates (0-1) to
                geographic coordinates (lat/lng)
              </Text>
            </Stack>
          </Card>
        </Collapse>
      </Stack>
    </Card>
  );
}
