// src/components/SettingsModal.tsx
"use client";

import React from 'react';

// Ho aggiornato l'interfaccia per ricevere 'userTier' come suggerito nel piano d'azione
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
  
  // LA PARENTESI SBAGLIATA E' STATA RIMOSSA DA QUI

  return (
    // Overlay di sfondo
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-60 backdrop-blur-sm"
      onClick={onClose}
    >
      {/* Contenitore del modale */}
      <div
        className="relative w-full max-w-lg rounded-xl bg-gray-800 p-8 shadow-2xl border border-gray-700"
        onClick={(e) => e.stopPropagation()} 
      >
        {/* Pulsante di chiusura */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors"
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        <h2 className="text-3xl font-bold text-blue-400 mb-6">Impostazioni</h2>

        {/* Sezione Motore AI */}
        <div className="mb-8">
          <label htmlFor="ai-engine" className="block text-lg font-semibold text-gray-300 mb-2">
            Motore di Elaborazione
          </label>
          <select
            id="ai-engine"
            disabled
            className="w-full rounded-lg border border-gray-600 bg-gray-700 p-3 text-gray-400 cursor-not-allowed focus:outline-none"
          >
            <option>Gemini (gemini-flash-lite)</option>
          </select>
          <p className="text-sm text-gray-500 mt-2">
            Stiamo lavorando per integrare nuovi motori AI in futuro.
          </p>
        </div>

        {/* Sezione Abbonamento */}
        <div>
          <h3 className="block text-lg font-semibold text-gray-300 mb-2">
            Abbonamento
          </h3>
          <div className="flex items-center justify-between rounded-lg bg-gray-700 p-4 border border-gray-600">
            <div>
              <p className="text-gray-400">Piano attuale:</p>
              {/* Usa la prop 'userTier' per mostrare il piano dinamicamente */}
              <p className="text-xl font-bold text-white capitalize">{userTier}</p>
            </div>
            <button
              onClick={handleStripeClick}
              className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg shadow-md transition-colors duration-200"
            >
              Gestisci Abbonamento
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}