// src/components/CTOVModal.tsx
"use client";

import React, { useState, useEffect } from 'react';
import { useAuth } from '@clerk/nextjs';
import { useUsage } from '@/context/UsageContext';

// Definiamo i tipi per le props del componente
interface CTOVProfileData {
  id?: string;
  name: string;
  mission?: string;       // Da 'string' a 'string | undefined'
  archetype?: string;     // Da 'string' a 'string | undefined'
  tone_traits?: string[]; // Da 'string[]' a 'string[] | undefined'
  banned_terms?: string[];// Da 'string[]' a 'string[] | undefined'
}

interface CTOVModalProps {
  isOpen: boolean;
  onClose: () => void;
  // Ora il tipo richiesto dalla prop è coerente con quello che gli passiamo
  profileToEdit?: CTOVProfileData | null;
}

const archetypes = ["The Sage (L'Esperto)", "The Ruler (Il Regnante)", "The Jester (Il Giullare)", "The Everyman (La Persona Comune)"];

export default function CTOVModal({ isOpen, onClose, profileToEdit }: CTOVModalProps) {
  const [formData, setFormData] = useState<Omit<CTOVProfileData, 'id'>>({
    name: '',
    mission: '',
    archetype: archetypes[0],
    tone_traits: [], // Inizializza SEMPRE come array vuoto
    banned_terms: [], // Inizializza SEMPRE come array vuoto
  });
  
  const [toneTraitInput, setToneTraitInput] = useState('');
  const [bannedTermInput, setBannedTermInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const { getToken } = useAuth();
  const { fetchUserStatus } = useUsage();

  // Se viene passato un profilo da modificare, pre-compiliamo il form
  useEffect(() => {
    if (profileToEdit) {
      setFormData({
        name: profileToEdit.name || '',
        mission: profileToEdit.mission || '',
        archetype: profileToEdit.archetype || archetypes[0],
        // Usiamo '|| []' per garantire che, se il dato è null o undefined,
        // lo stato venga impostato su un array vuoto.
        tone_traits: profileToEdit.tone_traits || [],
        banned_terms: profileToEdit.banned_terms || [],
      });
    } else {
      // Reset del form se il modale viene aperto per creare un nuovo profilo
      setFormData({ name: '', mission: '', archetype: archetypes[0], tone_traits: [], banned_terms: [] });
    }
  }, [profileToEdit, isOpen]);

  if (!isOpen) return null;

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const addToList = (listName: 'tone_traits' | 'banned_terms', value: string) => {
    if (value.trim() && !formData[listName]?.includes(value.trim())) { // Aggiunto '?' per sicurezza extra
      setFormData(prev => ({ ...prev, [listName]: [...(prev[listName] || []), value.trim()] })); // Aggiunto '|| []'
    }
  };

  const removeFromList = (listName: 'tone_traits' | 'banned_terms', valueToRemove: string) => {
    setFormData(prev => ({ ...prev, [listName]: (prev[listName] || []).filter(item => item !== valueToRemove) }));
  };
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.name) {
      setError("Il nome della Voce è obbligatorio.");
      return;
    }
    setIsLoading(true);
    setError(null);
    
    try {
        const token = await getToken();
        if (!token) throw new Error("Token non disponibile.");

        const isEditing = profileToEdit && profileToEdit.id;
        const url = isEditing
            ? `${process.env.NEXT_PUBLIC_API_URL}/ctov-profiles/${profileToEdit.id}`
            : `${process.env.NEXT_PUBLIC_API_URL}/ctov-profiles`;
        
        const method = isEditing ? 'PUT' : 'POST';

        const response = await fetch(url, {
            method: method,
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
            body: JSON.stringify(formData),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || "Si è verificato un errore durante il salvataggio.");
        }

        await fetchUserStatus();
        onClose();

    } catch (err) {
        setError(err instanceof Error ? err.message : "Errore sconosciuto.");
    } finally {
        setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-70 backdrop-blur-sm" onClick={onClose}>
      <div className="relative w-full max-w-2xl bg-gray-800 p-8 rounded-xl shadow-2xl border border-gray-700" onClick={(e) => e.stopPropagation()}>
        <h2 className="text-3xl font-bold text-blue-400 mb-6">{profileToEdit ? "Modifica Voce" : "Crea Nuova Voce Personalizzata"}</h2>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Nome della Voce */}
          <div>
            <label htmlFor="name" className="block text-sm font-medium text-gray-300 mb-1">Nome Voce</label>
            <input type="text" name="name" id="name" value={formData.name} onChange={handleInputChange} className="w-full input-style" placeholder="Es. Voce Marketing Q4" required />
          </div>

          {/* Missione */}
          <div>
            <label htmlFor="mission" className="block text-sm font-medium text-gray-300 mb-1">Missione del Brand (Max 250 caratteri)</label>
            <textarea name="mission" id="mission" value={formData.mission} onChange={handleInputChange} rows={2} className="w-full input-style" placeholder="Es. Aiutiamo le PMI a gestire la compliance..."></textarea>
          </div>

          {/* Archetipo */}
          <div>
            <label htmlFor="archetype" className="block text-sm font-medium text-gray-300 mb-1">Archetipo di Brand</label>
            <select name="archetype" id="archetype" value={formData.archetype} onChange={handleInputChange} className="w-full input-style">
              {archetypes.map(arch => <option key={arch} value={arch}>{arch}</option>)}
            </select>
          </div>
          
          {/* Tratti del Tono & Termini Proibiti (in due colonne) */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Tratti del Tono */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Tratti del Tono (Max 3)</label>
              <div className="flex">
                <input type="text" value={toneTraitInput} onChange={(e) => setToneTraitInput(e.target.value)} className="flex-grow input-style rounded-r-none" placeholder="Es. Esperto, Sicuro..." />
                <button type="button" onClick={() => { addToList('tone_traits', toneTraitInput); setToneTraitInput(''); }} className="btn-secondary rounded-l-none px-4">Aggiungi</button>
              </div>
              <div className="mt-2 space-x-2">
                {formData.tone_traits?.map(trait => (
                  <span key={trait} className="inline-flex items-center bg-blue-800 text-blue-200 text-sm font-medium px-2.5 py-0.5 rounded-full">
                    {trait}
                    <button type="button" onClick={() => removeFromList('tone_traits', trait)} className="ml-1.5 text-blue-300 hover:text-white">&times;</button>
                  </span>
                ))}
              </div>
            </div>
            {/* Termini Proibiti */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">Termini Proibiti</label>
              <div className="flex">
                <input type="text" value={bannedTermInput} onChange={(e) => setBannedTermInput(e.target.value)} className="flex-grow input-style rounded-r-none" placeholder="Es. garantito, rivoluzionario..." />
                <button type="button" onClick={() => { addToList('banned_terms', bannedTermInput); setBannedTermInput(''); }} className="btn-secondary rounded-l-none px-4">Aggiungi</button>
              </div>
              <div className="mt-2 space-x-2">
                 {formData.banned_terms?.map(term => (
                  <span key={term} className="inline-flex items-center bg-red-800 text-red-200 text-sm font-medium px-2.5 py-0.5 rounded-full">
                    {term}
                    <button type="button" onClick={() => removeFromList('banned_terms', term)} className="ml-1.5 text-red-300 hover:text-white">&times;</button>
                  </span>
                ))}
              </div>
            </div>
          </div>
          
          {error && <p className="text-red-400 text-sm text-center">{error}</p>}

          <div className="flex justify-end space-x-4 pt-4">
            <button type="button" onClick={onClose} disabled={isLoading} className="btn-secondary">Annulla</button>
            <button type="submit" disabled={isLoading} className="btn-primary">
              {isLoading ? 'Salvataggio...' : 'Salva Voce'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// Aggiungi queste classi di utility al tuo file `globals.css` per riutilizzare lo stile
/*
.input-style {
  @apply block w-full bg-gray-900 border border-gray-600 rounded-md shadow-sm py-2 px-3 text-gray-200 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm;
}
.btn-primary {
  @apply bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg shadow-md transition-colors disabled:bg-gray-700 disabled:cursor-not-allowed;
}
.btn-secondary {
  @apply bg-gray-600 hover:bg-gray-500 text-white font-semibold py-2 px-4 rounded-lg shadow-md transition-colors disabled:bg-gray-800;
}
*/