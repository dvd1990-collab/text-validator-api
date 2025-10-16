// src/context/UsageContext.tsx
"use client";

import React, { createContext, useState, useContext, ReactNode, useCallback } from 'react';
import { useAuth } from '@clerk/nextjs';

// Interfaccia per un singolo profilo CTOV (deve corrispondere al modello Pydantic)
interface CTOVProfile {
  id: string;
  name: string;
  mission?: string;
  archetype?: string;
  tone_traits?: string[];
  banned_terms?: string[];
}

interface UsageContextType {
  usageCount: number | null;
  usageLimit: number | null;
  userTier: string;
  validator_profiles: string[] | 'all';
  interpreter_profiles: string[] | 'all';
  complianceAccess: boolean;
  // --- NUOVI STATI PER CTOV ---
  ctovAccess: boolean;                // L'utente ha accesso alla feature?
  ctovMaxProfiles: number | null;     // Quanti profili può creare?
  ctovProfiles: CTOVProfile[];        // La lista dei suoi profili
  // --- FINE NUOVI STATI ---
  fetchUserStatus: () => Promise<void>;
}

const UsageContext = createContext<UsageContextType | undefined>(undefined);

export const UsageProvider = ({ children }: { children: ReactNode }) => {
  const [usageCount, setUsageCount] = useState<number | null>(null);
  const [usageLimit, setUsageLimit] = useState<number | null>(null);
  const [userTier, setUserTier] = useState<string>('free');
  const [validator_profiles, setValidatorProfiles] = useState<string[] | 'all'>([]);
  const [interpreter_profiles, setInterpreterProfiles] = useState<string[] | 'all'>([]);
  const [complianceAccess, setComplianceAccess] = useState<boolean>(false);
  
  // --- AGGIUNGI I NUOVI STATI PER CTOV ---
  const [ctovAccess, setCtovAccess] = useState<boolean>(false);
  const [ctovMaxProfiles, setCtovMaxProfiles] = useState<number | null>(0);
  const [ctovProfiles, setCtovProfiles] = useState<CTOVProfile[]>([]);
  // --- FINE AGGIUNTA ---

  const { getToken } = useAuth();

  const fetchUserStatus = useCallback(async () => {
    try {
      const token = await getToken();
      if (!token) return;
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/user-status`, { headers: { 'Authorization': `Bearer ${token}` } });
      if (!response.ok) throw new Error('Failed to fetch status');
      
      const data = await response.json();
      
      // Stati esistenti
      setUsageCount(data.usage.count);
      setUsageLimit(data.usage.limit);
      setUserTier(data.tier);
      setValidatorProfiles(data.validator_profiles);
      setInterpreterProfiles(data.interpreter_profiles);
	  setComplianceAccess(data.compliance_access);

      // --- AGGIORNA I NUOVI STATI CTOV CON I DATI DALL'API ---
      setCtovAccess(data.ctov_access);
      setCtovMaxProfiles(data.ctov_max_profiles);
      setCtovProfiles(data.ctov_profiles);
      // --- FINE AGGIORNAMENTO ---

    } catch (error) {
      console.error("Errore nel recupero dello stato utente (Context):", error);
    }
  }, [getToken]);

  // --- AGGIUNGI I NUOVI STATI AL VALORE DEL CONTEXT ---
  const value = { 
    usageCount, 
    usageLimit, 
    userTier, 
    validator_profiles, 
    interpreter_profiles, 
    complianceAccess,
    ctovAccess,
    ctovMaxProfiles,
    ctovProfiles,
    fetchUserStatus 
  };

  return (
    <UsageContext.Provider value={value}>
      {children}
    </UsageContext.Provider>
  );
};

export const useUsage = () => {
  const context = useContext(UsageContext);
  if (context === undefined) throw new Error('useUsage must be used within a UsageProvider');
  return context;
};