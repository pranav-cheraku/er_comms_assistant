'use client';

import { useState, useEffect } from 'react';
import { MicrophoneIcon } from '@heroicons/react/24/solid';
import { useReactMediaRecorder } from 'react-media-recorder';

type Message = {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
};

type Entity = {
  text: string;
  label: string;
  category: string;
};

type BackendResponse = {
  summary: string;
  entities: Entity[];
  original_text: string;
};

export default function ClientNurseChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isRecordingSupported, setIsRecordingSupported] = useState(true);

  const {
    status,
    startRecording,
    stopRecording,
    mediaBlobUrl,
    error
  } = useReactMediaRecorder({
    audio: true,
    blobPropertyBag: { type: 'audio/webm' },
    onError: (err) => {
      console.error('Recording error:', err);
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: 'There was an error with the recording. Please try again.',
          timestamp: new Date().toLocaleTimeString()
        }
      ]);
    },
    onStop: async (blobUrl, blob) => {
      if (!blob) {
        console.error('No blob received from recording');
        setMessages(prev => [
          ...prev,
          {
            role: 'assistant',
            content: 'Failed to record audio. Please try again.',
            timestamp: new Date().toLocaleTimeString()
          }
        ]);
        return;
      }

      setIsProcessing(true);
      try {
        // Generate timestamp for filename
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `recording_${timestamp}.webm`;
        
        // Convert blob to File object
        const audioFile = new File([blob], filename, { type: 'audio/webm' });
        
        // Create FormData and append the audio file
        const formData = new FormData();
        formData.append('audio', audioFile);
        formData.append('filename', filename);

        // Send to backend for processing
        const response = await fetch('http://localhost:8000/api/process-audio', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Failed to process audio');
        }

        const data = await response.json();
        
        // Add the transcribed and summarized text to messages with timestamp
        setMessages(prev => [
          ...prev,
          { 
            role: 'user', 
            content: `Transcription: ${data.transcription}\nSummary: ${data.summary}`,
            timestamp: new Date().toLocaleTimeString()
          }
        ]);

        // Clear the recording URL after successful processing
        if (mediaBlobUrl) {
          URL.revokeObjectURL(mediaBlobUrl);
        }
      } catch (error) {
        console.error('Error processing audio:', error);
        setMessages(prev => [
          ...prev,
          { 
            role: 'assistant', 
            content: `Error: ${error instanceof Error ? error.message : 'Failed to process recording'}. Please try again.`,
            timestamp: new Date().toLocaleTimeString()
          }
        ]);
      } finally {
        setIsProcessing(false);
      }
    },
  });

  // Log any recording errors
  useEffect(() => {
    if (error) {
      console.error('Recording error:', error);
    }
  }, [error]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const timestamp = new Date().toLocaleTimeString();
    // Add user message
    const newMessages: Message[] = [...messages, { 
      role: 'user', 
      content: input,
      timestamp 
    }];
    setMessages(newMessages);
    setInput('');

    try {
      // Send text to backend for processing
      const response = await fetch('http://localhost:8000/api/process-text', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: input }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to process message');
      }

      const data: BackendResponse = await response.json();
      
      // Add the assistant's response to messages
      setMessages([...newMessages, { 
        role: 'assistant', 
        content: `Summary: ${data.summary}\n\nKey Medical Terms: ${data.entities.map(e => e.text).join(', ')}`,
        timestamp: new Date().toLocaleTimeString()
      }]);
    } catch (error) {
      console.error('Error processing message:', error);
      setMessages([...newMessages, { 
        role: 'assistant', 
        content: `Error: ${error instanceof Error ? error.message : 'Failed to process message'}. Please try again.`,
        timestamp: new Date().toLocaleTimeString()
      }]);
    }
  };

  const handleRecord = async () => {
    try {
      if (status === 'recording') {
        stopRecording();
      } else {
        await startRecording();
      }
    } catch (err) {
      console.error('Error handling recording:', err);
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: 'Failed to start recording. Please try again.',
          timestamp: new Date().toLocaleTimeString()
        }
      ]);
    }
  };

  return (
    <div className="h-[600px] flex flex-col">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto mb-6 space-y-6">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className="flex flex-col">
              <div
                className={`max-w-[80%] rounded-lg p-4 ${
                  message.role === 'user'
                    ? 'bg-emerald-600 text-white'
                    : 'bg-slate-50 text-slate-800 border border-slate-200'
                }`}
              >
                {message.role === 'user' && message.content.includes('Transcription:') ? (
                  <div className="space-y-2">
                    <div className="font-medium">Transcription:</div>
                    <div className="text-sm">{message.content.split('Transcription:')[1].split('Summary:')[0].trim()}</div>
                    <div className="font-medium mt-2">Summary:</div>
                    <div className="text-sm">{message.content.split('Summary:')[1].trim()}</div>
                  </div>
                ) : message.role === 'assistant' && message.content.includes('Summary:') ? (
                  <div className="space-y-2">
                    <div className="font-medium">Summary:</div>
                    <div className="text-sm">{message.content.split('Summary:')[1].split('Key Medical Terms:')[0].trim()}</div>
                    <div className="font-medium mt-2">Key Medical Terms:</div>
                    <div className="text-sm">{message.content.split('Key Medical Terms:')[1].trim()}</div>
                  </div>
                ) : (
                  message.content
                )}
              </div>
              {message.timestamp && (
                <span className="text-xs text-slate-500 mt-1">
                  {message.timestamp}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="flex gap-3">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your medical question here..."
          className="flex-1 p-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500 text-slate-800 bg-white"
        />
        <button
          type="button"
          onClick={handleRecord}
          disabled={isProcessing}
          className={`p-3 rounded-lg transition-colors flex items-center gap-2 ${
            status === 'recording'
              ? 'bg-red-500 hover:bg-red-600'
              : isProcessing
              ? 'bg-slate-400 cursor-not-allowed'
              : 'bg-emerald-600 hover:bg-emerald-700'
          }`}
          title={status === 'recording' ? "Stop Recording" : "Start Recording"}
        >
          <MicrophoneIcon className="h-6 w-6 text-white" />
          <span className="text-white font-medium">
            {status === 'recording' ? 'Stop Recording' : 'Start Transcribing'}
          </span>
        </button>
        <button
          type="submit"
          className="bg-emerald-600 text-white px-6 py-3 rounded-lg hover:bg-emerald-700 transition-colors font-medium"
        >
          Send
        </button>
      </form>

      {/* Recording Status */}
      {status === 'recording' && (
        <div className="mt-2 text-sm text-red-500 flex items-center gap-2">
          <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
          Recording...
        </div>
      )}
      {isProcessing && (
        <div className="mt-2 text-sm text-slate-500">
          Processing your recording...
        </div>
      )}
      {error && (
        <div className="mt-2 text-sm text-red-500">
          Error: {error}
        </div>
      )}
    </div>
  );
} 