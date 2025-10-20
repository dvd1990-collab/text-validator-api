// src/components/Navbar.tsx
"use client";

import Link from 'next/link';
import { useUser, useAuth } from '@clerk/nextjs';
import { useRouter } from 'next/navigation';
import { useState, useEffect } from 'react';
import GearIcon from './GearIcon';
import SettingsModal from './SettingsModal';
import { useUsage } from '@/context/UsageContext'; // Importa il nostro hook personalizzato

export default function Navbar() {
  const { isLoaded, isSignedIn, user } = useUser();
  const { signOut } = useAuth(); // Prendiamo solo 'signOut' da qui
  const router = useRouter();
  
  // Utilizza i dati e le funzioni direttamente dal Context.
  // NON ci sono più stati locali per usageCount, usageLimit, userTier.
  const { usageCount, usageLimit, userTier, fetchUserStatus } = useUsage();

  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // Questo useEffect serve solo a caricare i dati la prima volta che l'utente è autenticato.
  useEffect(() => {
    if (isLoaded && isSignedIn) {
      fetchUserStatus();
    }
  }, [isLoaded, isSignedIn, fetchUserStatus]);

  const handleLogout = async () => {
    try {
      // Al logout, reindirizziamo al login
      await signOut(() => router.push('/login'));
    } catch (error) {
      console.error('Errore durante il logout:', error);
    }
  };

  return (
    <>
      <nav className="fixed w-full z-40 bg-gray-900 border-b border-gray-700 shadow-lg py-4 px-8 md:px-12 flex justify-between items-center">
        <div className="flex items-center">
          <Link href="/text-validator-ai" className="text-3xl font-extrabold text-blue-400 hover:text-blue-300 transition-colors duration-200">
            Text Validator
          </Link>
        </div>

        {/* Menu Hamburger per mobile */}
        <div className="md:hidden">
          <button onClick={() => setIsMenuOpen(!isMenuOpen)} className="text-white focus:outline-none">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={isMenuOpen ? "M6 18L18 6M6 6l12 12" : "M4 6h16M4 12h16M4 18h16"}></path>
            </svg>
          </button>
        </div>

        {/* Links e Bottoni per desktop e menu mobile */}
        <div className={`${isMenuOpen ? 'flex' : 'hidden'} md:flex flex-col md:flex-row md:items-center space-y-4 md:space-y-0 md:space-x-8 absolute md:static top-full left-0 w-full md:w-auto bg-gray-900 md:bg-transparent p-8 md:p-0 border-t md:border-t-0 border-gray-700 shadow-lg md:shadow-none`}>
          <Link href="/strategist" className="text-gray-300 hover:text-purple-300 text-lg font-semibold transition-colors duration-200">Strategist</Link>
		  <Link href="/" className="text-gray-300 hover:text-blue-300 text-lg font-semibold transition-colors duration-200">Validator</Link>
		  <Link href="/interpreter" className="text-gray-300 hover:text-blue-300 text-lg font-semibold transition-colors duration-200">Interpreter</Link>
		  <Link href="/compliance-checkr" className="text-gray-300 hover:text-blue-300 text-lg font-semibold transition-colors duration-200">Compliance Checkr</Link>
          <Link href="/roadmap" className="text-gray-300 hover:text-blue-300 text-lg font-semibold transition-colors duration-200">Roadmap</Link>
          <Link href="/use-cases" className="text-gray-300 hover:text-blue-300 text-lg font-semibold transition-colors duration-200">Use Cases</Link>
          <Link href="/pricing" className="text-gray-300 hover:text-blue-300 text-lg font-semibold transition-colors duration-200">Pricing</Link>
          {!isLoaded ? (
            <span className="text-gray-500">Caricamento...</span>
          ) : !isSignedIn ? (
            <div className="flex flex-col md:flex-row space-y-2 md:space-y-0 md:space-x-4">
              <Link href="/login" className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors duration-200 shadow-md">Login</Link>
              <Link href="/sign-up" className="bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors duration-200 shadow-md">Registrati</Link>
            </div>
          ) : (
            <div className="flex flex-col md:flex-row md:items-center space-y-4 md:space-y-0 md:space-x-4">
              <button onClick={() => setIsSettingsOpen(true)} className="text-gray-400 hover:text-white transition-colors">
                <GearIcon className="w-6 h-6" />
              </button>

              {/* Contatore che legge i dati direttamente dal Context */}
              {usageCount !== null && usageLimit !== null && (
                <div className="text-sm text-gray-300 bg-gray-700 px-3 py-1 rounded-md shadow-sm">
                    {usageLimit === -1 
                        ? (<span>Utilizzo: Illimitato</span>) 
                        : (<span>Chiamate: {usageCount} / {usageLimit}</span>)
                    }
                </div>
              )}

              <button onClick={handleLogout} className="bg-red-700 hover:bg-red-600 text-white font-semibold py-2 px-4 rounded-lg transition-colors duration-200 shadow-md">
                Logout ({user?.emailAddresses[0]?.emailAddress || 'utente'})
              </button>
            </div>
          )}
        </div>
      </nav>

      {/* Il modale riceve 'userTier' direttamente dal Context */}
      <SettingsModal isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} userTier={userTier} />
    </>
  );
}