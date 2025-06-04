'use client';

import { useState, useEffect } from 'react';

type ClientRecord = {
  id: string;
  name: string;
  timestamp: string;
  transcription: string;
  summary: string;
};

export default function DoctorNurseChat() {
  const [records, setRecords] = useState<ClientRecord[]>([]);
  const [selectedRecord, setSelectedRecord] = useState<ClientRecord | null>(null);
  const [searchTerm, setSearchTerm] = useState('');

  // TODO: Replace with actual API call
  useEffect(() => {
    // Mock data for demonstration
    const mockRecords: ClientRecord[] = [
      {
        id: '1',
        name: 'John Doe',
        timestamp: '2024-03-20 10:30 AM',
        transcription: 'Patient reported severe headache and fever for the past 24 hours.',
        summary: 'Patient presenting with acute headache and fever. Symptoms started 24 hours ago.'
      },
      {
        id: '2',
        name: 'Jane Smith',
        timestamp: '2024-03-20 11:15 AM',
        transcription: 'Patient experiencing chest pain and shortness of breath.',
        summary: 'Patient reporting chest pain and dyspnea. Immediate attention required.'
      }
    ];
    setRecords(mockRecords);
  }, []);

  const filteredRecords = records.filter(record =>
    record.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    record.transcription.toLowerCase().includes(searchTerm.toLowerCase()) ||
    record.summary.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="h-[600px] flex gap-6">
      {/* Records List */}
      <div className="w-1/3 flex flex-col">
        <div className="mb-4">
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search records..."
            className="w-full p-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500"
          />
        </div>
        <div className="flex-1 overflow-y-auto space-y-2">
          {filteredRecords.map((record) => (
            <button
              key={record.id}
              onClick={() => setSelectedRecord(record)}
              className={`w-full text-left p-4 rounded-lg transition-colors ${
                selectedRecord?.id === record.id
                  ? 'bg-emerald-50 border-2 border-emerald-500'
                  : 'bg-white border border-slate-200 hover:border-emerald-300'
              }`}
            >
              <div className="font-medium text-slate-800">{record.name}</div>
              <div className="text-sm text-slate-500">{record.timestamp}</div>
              <div className="text-sm text-slate-600 mt-1 line-clamp-2">
                {record.summary}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Record Details */}
      <div className="flex-1 bg-white rounded-lg border border-slate-200 p-6 overflow-y-auto">
        {selectedRecord ? (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-semibold text-slate-800">{selectedRecord.name}</h2>
              <p className="text-slate-500">{selectedRecord.timestamp}</p>
            </div>
            
            <div>
              <h3 className="text-lg font-medium text-slate-800 mb-2">Summary</h3>
              <p className="text-slate-600 bg-slate-50 p-4 rounded-lg">
                {selectedRecord.summary}
              </p>
            </div>

            <div>
              <h3 className="text-lg font-medium text-slate-800 mb-2">Full Transcription</h3>
              <p className="text-slate-600 bg-slate-50 p-4 rounded-lg">
                {selectedRecord.transcription}
              </p>
            </div>
          </div>
        ) : (
          <div className="h-full flex items-center justify-center text-slate-500">
            Select a record to view details
          </div>
        )}
      </div>
    </div>
  );
} 