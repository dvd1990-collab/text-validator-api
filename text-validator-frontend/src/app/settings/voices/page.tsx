// src/app/settings/voices/page.tsx
"use client";

import { useState } from 'react';
import { useUsage } from '@/context/UsageContext';
import { useAuth } from '@clerk/nextjs';
import CTOVModal from '@/components/CTOVModal'; // Riutilizziamo il modale

// Definiamo di nuovo il tipo qui per usarlo nello stato locale
interface CTOVProfile {
  id: string;
  name: string;
  mission: string;
  archetype: string;
  tone_traits: string[];
  banned_terms: string[];
}

export default function ManageVoicesPage() {
  const { ctovProfiles, fetchUserStatus } = useUsage();
  const { getToken } = useAuth();
  
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [profileToEdit, setProfileToEdit] = useState<CTOVProfile | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleEdit = (profile: CTOVProfile) => {
    setProfileToEdit(profile);
    setIsModalOpen(true);
  };

  const handleDelete = async (profileId: string, profileName: string) => {
    if (window.confirm(`Sei sicuro di voler eliminare la voce "${profileName}"?`)) {
      setError(null);
      try {
        const token = await getToken();
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/ctov-profiles/${profileId}`, {
          method: 'DELETE',
          headers: { 'Authorization': `Bearer ${token}` },
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || "Impossibile eliminare il profilo.");
        }
        
        await fetchUserStatus(); // Aggiorna la lista
      } catch (err) {
        setError(err instanceof Error ? err.message : "Errore sconosciuto.");
      }
    }
  };

  return (
    <>
      <CTOVModal isOpen={isModalOpen} onClose={() => { setIsModalOpen(false); setProfileToEdit(null); }} profileToEdit={profileToEdit} />
      
      <main className="flex flex-col items-center bg-gray-900 p-8 text-white">
        <div className="w-full max-w-4xl">
          <header className="mb-12 text-center">
            <h1 className="text-4xl font-extrabold text-blue-400">Gestione Voci Personalizzate</h1>
            <p className="mt-2 text-lg text-gray-400">Modifica ed elimina le tue voci CTOV.</p>
          </header>

          {error && <p className="text-red-400 text-center mb-4">{error}</p>}
          
          <div className="space-y-4">
            {ctovProfiles.length > 0 ? (
              ctovProfiles.map((profile) => (
                <div key={profile.id} className="bg-gray-800 p-4 rounded-lg flex justify-between items-center border border-gray-700">
                  <p className="text-xl font-semibold text-white">{profile.name}</p>
                  <div className="space-x-2">
                    <button onClick={() => handleEdit(profile)} className="btn-secondary px-4 py-2">Modifica</button>
                    <button onClick={() => handleDelete(profile.id, profile.name)} className="bg-red-700 hover:bg-red-600 text-white font-semibold py-2 px-4 rounded-lg shadow-md transition-colors">Elimina</button>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-center text-gray-500">Non hai ancora creato nessuna voce personalizzata.</p>
            )}
          </div>
        </div>
      </main>
    </>
  );
}