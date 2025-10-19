import { Text } from '@mantine/core';

interface MarkdownRendererProps {
  content: string;
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
}

export const MarkdownRenderer = ({
  content,
  size = 'sm',
}: MarkdownRendererProps) => {
  // Simple markdown-like rendering for basic formatting
  const renderMarkdown = (text: string) => {
    // Split by lines and process each line
    const lines = text.split('\n');
    const processedLines = lines.map((line, index) => {
      // Headers
      if (line.startsWith('## ')) {
        return (
          <Text key={index} fw={600} size='lg' mt='md' mb='sm'>
            {line.replace('## ', '')}
          </Text>
        );
      }
      if (line.startsWith('### ')) {
        return (
          <Text key={index} fw={600} size='md' mt='sm' mb='xs'>
            {line.replace('### ', '')}
          </Text>
        );
      }
      if (line.startsWith('# ')) {
        return (
          <Text key={index} fw={700} size='xl' mt='lg' mb='md'>
            {line.replace('# ', '')}
          </Text>
        );
      }

      // Bold text
      if (line.startsWith('**') && line.endsWith('**')) {
        return (
          <Text key={index} fw={600} size={size}>
            {line.replace(/\*\*/g, '')}
          </Text>
        );
      }

      // List items
      if (line.startsWith('- ') || line.startsWith('* ')) {
        return (
          <Text key={index} size={size} style={{ marginLeft: '16px' }}>
            • {line.replace(/^[-*] /, '')}
          </Text>
        );
      }

      // Numbered list items
      if (/^\d+\. /.test(line)) {
        return (
          <Text key={index} size={size} style={{ marginLeft: '16px' }}>
            {line}
          </Text>
        );
      }

      // Empty lines
      if (line.trim() === '') {
        return <br key={index} />;
      }

      // Regular text
      return (
        <Text key={index} size={size} style={{ marginBottom: '4px' }}>
          {line}
        </Text>
      );
    });

    return processedLines;
  };

  return <div>{renderMarkdown(content)}</div>;
};
