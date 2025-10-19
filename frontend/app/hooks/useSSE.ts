import { useRef, useCallback, useEffect } from 'react';

export interface SSEEvent {
  type: string;
  data: any;
  timestamp?: string;
}

export interface UseSSEOptions {
  onMessage?: (event: SSEEvent) => void;
  onError?: (error: Event) => void;
  onOpen?: () => void;
  onClose?: () => void;
}

export function useSSE() {
  const eventSourceRef = useRef<EventSource | null>(null);
  const expectedCloseRef = useRef<boolean>(false);

  const connect = useCallback((url: string, options: UseSSEOptions = {}) => {
    // Close existing connection if any
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    // Reset expected close flag
    expectedCloseRef.current = false;

    // Get auth token for SSE connection
    const token = localStorage.getItem('access_token');
    const urlWithAuth = token
      ? `${url}?token=${encodeURIComponent(token)}`
      : url;

    const eventSource = new EventSource(urlWithAuth, {
      withCredentials: true,
    });

    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      console.log('SSE connection opened');
      options.onOpen?.();
    };

    // Generic message handler (for events without custom type)
    eventSource.onmessage = (event) => {
      try {
        console.log('SSE message received:', event);

        // Check if event.data exists and is not empty
        if (
          !event.data ||
          typeof event.data !== 'string' ||
          event.data.trim() === ''
        ) {
          console.warn('SSE message event has no data, skipping');
          return;
        }

        const data = JSON.parse(event.data);
        const sseEvent: SSEEvent = {
          type: 'message',
          data,
          timestamp: new Date().toISOString(),
        };
        console.log('Parsed SSE event:', sseEvent);
        options.onMessage?.(sseEvent);
      } catch (error) {
        console.error(
          'Failed to parse SSE event:',
          error,
          'Raw data:',
          event.data
        );
      }
    };

    // Handler for custom event types
    const handleCustomEvent = (eventType: string) => (event: MessageEvent) => {
      try {
        console.log(`SSE ${eventType} event received:`, event);

        // Skip processing if we're already expecting the connection to close
        // This prevents processing spurious error events during shutdown
        if (expectedCloseRef.current && eventType === 'error') {
          console.log('Ignoring error event during expected connection close');
          return;
        }

        // Check if event.data exists and is not empty
        if (
          !event.data ||
          typeof event.data !== 'string' ||
          event.data.trim() === ''
        ) {
          console.warn(`SSE ${eventType} event has no data, skipping`);
          return;
        }

        const data = JSON.parse(event.data);
        const sseEvent: SSEEvent = {
          type: eventType,
          data,
          timestamp: new Date().toISOString(),
        };
        console.log('Parsed SSE event:', sseEvent);
        options.onMessage?.(sseEvent);

        // Mark connection as expected to close for terminal events
        if (eventType === 'report_end' || eventType === 'error') {
          expectedCloseRef.current = true;
          // Give a small delay for any remaining events, then close
          setTimeout(() => {
            if (eventSourceRef.current) {
              console.log(`Closing SSE connection after ${eventType} event`);
              eventSourceRef.current.close();
              eventSourceRef.current = null;
              options.onClose?.();
            }
          }, 100);
        }
      } catch (error) {
        console.error(
          `Failed to parse SSE ${eventType} event:`,
          error,
          'Raw data:',
          event.data
        );
      }
    };

    // Register listeners for all expected custom event types
    const eventTypes = [
      'connected',
      'thinking_start',
      'thinking_content',
      'thinking_end',
      'tool_call_start',
      'tool_call_result',
      'report_start',
      'report_content',
      'report_end',
      'collision_detected',
      'iteration_update',
      'model_switch',
      'error',
    ];

    eventTypes.forEach((eventType) => {
      eventSource.addEventListener(eventType, handleCustomEvent(eventType));
    });

    eventSource.onerror = (error) => {
      // Only treat as error if not an expected close
      if (!expectedCloseRef.current) {
        console.error('SSE connection error:', error);
        options.onError?.(error);
      } else {
        console.log('SSE connection closed as expected');
      }
      // Close connection on error
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };

    return eventSource;
  }, []);

  const disconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }, []);

  const isConnected = useCallback(() => {
    return eventSourceRef.current?.readyState === EventSource.OPEN;
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    connect,
    disconnect,
    isConnected,
  };
}
