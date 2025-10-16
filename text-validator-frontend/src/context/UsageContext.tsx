// src/context/UsageContext.tsx
"use client";

import React, { createContext, useState, useContext, ReactNode, useCallback } from 'react';
import { useAuth } from '@clerk/nextjs';

interface UsageContextType {
  usageCount: number | null;
  usageLimit: number | null;
  userTier: string;
  validator_profiles: string[] | 'all';
  interpreter_profiles: string[] | 'all';
  fetchUserStatus: () => Promise<void>;
  complianceAccess: boolean;
}

const UsageContext = createContext<UsageContextType | undefined>(undefined);

export const UsageProvider = ({ children }: { children: ReactNode }) => {
  const [usageCount, setUsageCount] = useState<number | null>(null);
  const [usageLimit, setUsageLimit] = useState<number | null>(null);
  const [userTier, setUserTier] = useState<string>('free');
  const [validator_profiles, setValidatorProfiles] = useState<string[] | 'all'>(['Generico', 'L\'Umanizzatore']);
  const [interpreter_profiles, setInterpreterProfiles] = useState<string[] | 'all'>(['Spiega in Parole Semplici']);
  const [complianceAccess, setComplianceAccess] = useState<boolean>(false);

  const { getToken } = useAuth();

  const fetchUserStatus = useCallback(async () => {
    // ... la funzione fetchUserStatus rimane identica, ma ora imposterà i nuovi stati ...
    try {
      const token = await getToken();
      if (!token) return;
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/user-status`, { headers: { 'Authorization': `Bearer ${token}` } });
      if (!response.ok) throw new Error('Failed to fetch status');
      const data = await response.json();
      setUsageCount(data.usage.count);
      setUsageLimit(data.usage.limit);
      setUserTier(data.tier);
      setValidatorProfiles(data.validator_profiles);
      setInterpreterProfiles(data.interpreter_profiles);
	  setComplianceAccess(data.compliance_access);
    } catch (error) {
      console.error("Errore nel recupero dello stato utente (Context):", error);
    }
  }, [getToken]);

  const value = { usageCount, usageLimit, userTier, validator_profiles, interpreter_profiles, fetchUserStatus, complianceAccess };

  return (
    <UsageContext.Provider value={value}>
      {children}
    </UsageContext.Provider>
  );
};

export const useUsage = () => {
  // ... la funzione useUsage rimane identica ...
  const context = useContext(UsageContext);
  if (context === undefined) throw new Error('useUsage must be used within a UsageProvider');
  return context;
};