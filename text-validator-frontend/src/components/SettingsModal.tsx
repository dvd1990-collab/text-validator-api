// src/components/SettingsModal.tsx
"use client";

import React from 'react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  userTier: string; 
}

export default function SettingsModal({ isOpen, onClose, userTier }: SettingsModalProps) {
  if (!isOpen) return null;

  const handleStripeClick = () => {
    alert("La gestione dell'abbonamento tramite Stripe sarà disponibile a breve.");
  };

  // Logica per determinare quale modello mostrare
  const validatorEngine = "Gemini Flash Lite";
  const interpreterEngine = (userTier === 'free') ? "Gemini Flash Lite" : "Gemini 2.5 Flash";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-60 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-lg rounded-xl bg-gray-800 p-8 shadow-2xl border border-gray-700"
        onClick={(e) => e.stopPropagation()} 
      >
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors"
        >
          {/* ... SVG icona chiusura ... */}
        </button>

        <h2 className="text-3xl font-bold text-blue-400 mb-6">Impostazioni</h2>

        {/* Sezione Motore AI */}
        <div className="mb-8">
          <h3 className="block text-lg font-semibold text-gray-300 mb-2">
            Motori di Elaborazione Attivi
          </h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between rounded-lg bg-gray-700 p-3 border border-gray-600">
                <p className="font-semibold text-gray-200">Validator Engine:</p>
                <p className="font-mono text-sm text-blue-300 bg-gray-800 px-2 py-1 rounded">{validatorEngine}</p>
            </div>
            <div className="flex items-center justify-between rounded-lg bg-gray-700 p-3 border border-gray-600">
                <p className="font-semibold text-gray-200">Interpreter Engine:</p>
                <p className="font-mono text-sm text-blue-300 bg-gray-800 px-2 py-1 rounded">{interpreterEngine}</p>
            </div>
			 <div className="flex items-center justify-between rounded-lg bg-gray-700 p-3 border border-gray-600">
            <p className="font-semibold text-gray-200">Compliance Engine:</p>
            <p className="font-mono ...">Gemini 2.5 Flash</p>
        </div>
          </div>
        </div>

        {/* Sezione Abbonamento */}
        <div>
          <h3 className="block text-lg font-semibold text-gray-300 mb-2">
            Abbonamento
          </h3>
          <div className="flex items-center justify-between rounded-lg bg-gray-700 p-4 border border-gray-600">
            <div>
              <p className="text-gray-400">Piano attuale:</p>
              <p className="text-xl font-bold text-white capitalize">{userTier}</p>
            </div>
            <button
              onClick={handleStripeClick}
              className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg shadow-md transition-colors"
            >
              Gestisci Abbonamento
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}