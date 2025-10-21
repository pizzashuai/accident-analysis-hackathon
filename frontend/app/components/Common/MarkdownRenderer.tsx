import { Text } from '@mantine/core';
import { Fragment, ReactNode } from 'react';

interface MarkdownRendererProps {
  content: string;
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
}

export const MarkdownRenderer = ({
  content,
  size = 'sm',
}: MarkdownRendererProps) => {
  const renderInlineMarkdown = (
    text: string,
    keyPrefix: string,
  ): ReactNode => {
    if (!text.includes('**')) {
      return text;
    }

    const segments = text.split(/(\*\*[^*]+\*\*)/g).filter(Boolean);

    return segments.map((segment, segmentIndex) => {
      if (segment.startsWith('**') && segment.endsWith('**') && segment.length > 4) {
        return (
          <strong key={`${keyPrefix}-bold-${segmentIndex}`}>
            {segment.slice(2, -2)}
          </strong>
        );
      }

      return (
        <Fragment key={`${keyPrefix}-text-${segmentIndex}`}>
          {segment}
        </Fragment>
      );
    });
  };

  // Simple markdown-like rendering for basic formatting
  const renderMarkdown = (text: string) => {
    // Split by lines and process each line
    const lines = text.split('\n');
    const processedLines = lines.map((line, index) => {
      // Headers
      if (line.startsWith('## ')) {
        return (
          <Text key={index} fw={600} size='lg' mt='md' mb='sm'>
            {renderInlineMarkdown(line.replace('## ', ''), `h2-${index}`)}
          </Text>
        );
      }
      if (line.startsWith('### ')) {
        return (
          <Text key={index} fw={600} size='md' mt='sm' mb='xs'>
            {renderInlineMarkdown(line.replace('### ', ''), `h3-${index}`)}
          </Text>
        );
      }
      if (line.startsWith('# ')) {
        return (
          <Text key={index} fw={700} size='xl' mt='lg' mb='md'>
            {renderInlineMarkdown(line.replace('# ', ''), `h1-${index}`)}
          </Text>
        );
      }

      // List items
      if (line.startsWith('- ') || line.startsWith('* ')) {
        return (
          <Text key={index} size={size} style={{ marginLeft: '16px' }}>
            • {renderInlineMarkdown(line.replace(/^[-*] /, ''), `list-${index}`)}
          </Text>
        );
      }

      // Numbered list items
      if (/^\d+\. /.test(line)) {
        return (
          <Text key={index} size={size} style={{ marginLeft: '16px' }}>
            {renderInlineMarkdown(line, `ordered-${index}`)}
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
          {renderInlineMarkdown(line, `text-${index}`)}
        </Text>
      );
    });

    return processedLines;
  };

  return <div>{renderMarkdown(content)}</div>;
};
