// src/context/UsageContext.tsx
"use client";

import React, { createContext, useState, useContext, ReactNode, useCallback } from 'react';
import { useAuth } from '@clerk/nextjs';

interface UsageContextType {
  usageCount: number | null;
  usageLimit: number | null;
  userTier: string;
  allowedProfiles: string[] | 'all';
  fetchUserStatus: () => Promise<void>; // Funzione per forzare l'aggiornamento
}

const UsageContext = createContext<UsageContextType | undefined>(undefined);

export const UsageProvider = ({ children }: { children: ReactNode }) => {
  const [usageCount, setUsageCount] = useState<number | null>(null);
  const [usageLimit, setUsageLimit] = useState<number | null>(null);
  const [userTier, setUserTier] = useState<string>('free');
  const [allowedProfiles, setAllowedProfiles] = useState<string[] | 'all'>(['Generico', 'L\'Umanizzatore']);

  const { getToken } = useAuth();

  const fetchUserStatus = useCallback(async () => {
    try {
      const token = await getToken();
      if (!token) return;

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/user-status`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (!response.ok) throw new Error('Failed to fetch status');
      
      const data = await response.json();
      setUsageCount(data.usage.count);
      setUsageLimit(data.usage.limit);
      setUserTier(data.tier);
      setAllowedProfiles(data.allowed_profiles);
    } catch (error) {
      console.error("Errore nel recupero dello stato utente (Context):", error);
    }
  }, [getToken]);

  const value = { usageCount, usageLimit, userTier, allowedProfiles, fetchUserStatus };

  return (
    <UsageContext.Provider value={value}>
      {children}
    </UsageContext.Provider>
  );
};

export const useUsage = () => {
  const context = useContext(UsageContext);
  if (context === undefined) {
    throw new Error('useUsage must be used within a UsageProvider');
  }
  return context;
};