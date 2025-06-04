'use client';

import { useState } from 'react';
import ClientNurseChat from './components/ClientNurseChat';
import DoctorNurseChat from './components/DoctorNurseChat';

export default function Home() {
  const [activeTab, setActiveTab] = useState<'client' | 'doctor'>('client');

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="max-w-4xl mx-auto p-4">
        {/* Header */}
        <header className="bg-white shadow-sm rounded-lg p-6 mb-6 border-b border-slate-200">
          <h1 className="text-3xl font-semibold text-slate-800">Medical Assistant</h1>
          <p className="text-slate-600 mt-2">Your AI-powered healthcare companion</p>
        </header>

        {/* Tabs */}
        <div className="bg-white rounded-lg shadow-sm mb-6 border border-slate-200">
          <div className="flex border-b border-slate-200">
            <button
              className={`flex-1 py-4 px-6 text-center font-medium ${
                activeTab === 'client'
                  ? 'text-emerald-600 border-b-2 border-emerald-600'
                  : 'text-slate-600 hover:text-slate-800'
              }`}
              onClick={() => setActiveTab('client')}
            >
              Client-Nurse Chat
            </button>
            <button
              className={`flex-1 py-4 px-6 text-center font-medium ${
                activeTab === 'doctor'
                  ? 'text-emerald-600 border-b-2 border-emerald-600'
                  : 'text-slate-600 hover:text-slate-800'
              }`}
              onClick={() => setActiveTab('doctor')}
            >
              Doctor-Nurse Records
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6 border border-slate-200">
          {activeTab === 'client' ? (
            <ClientNurseChat />
          ) : (
            <DoctorNurseChat />
          )}
        </div>

        {/* Disclaimer */}
        <div className="text-sm text-slate-500 text-center bg-slate-50 p-4 rounded-lg border border-slate-200">
          This is an AI assistant. Please consult with healthcare professionals for medical advice.
        </div>
      </div>
    </div>
  );
}
