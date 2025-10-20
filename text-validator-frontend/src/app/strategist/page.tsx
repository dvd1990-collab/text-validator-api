// src/app/strategist/page.tsx
"use client";

import { useState } from 'react';
import { useAuth } from '@clerk/nextjs';
import { useRouter } from 'next/navigation';
import { useUsage } from '@/context/UsageContext';
import ReactMarkdown from 'react-markdown';

export default function StrategistPage() {
    const [inputText, setInputText] = useState('');
    const [outputText, setOutputText] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [copyButtonText, setCopyButtonText] = useState('Copia');
    
    // Recuperiamo l'accesso specifico per questa feature dal nostro Context
    const { strategistAccess, fetchUserStatus } = useUsage();
    const router = useRouter();
    
    const strategistProfileGroups = {
        "Business & Corporate Strategy": [
            "Sviluppatore di Business Model Canvas",
            "Analista di Vantaggio Competitivo (Porter's Five Forces)",
            "Generatore di Mission & Vision Statement",
            "Strutturatore di Pitch per Investitori (Outline)"
        ],
        "Marketing & Product Development": [
            "Pianificatore di Strategia Go-to-Market (GTM)",
            "Ideatore di Pillar Page e Content Cluster",
            "Sviluppatore di Buyer Persona",
            "Generatore di Brief Creativo per Campagne",
            "Strutturatore di Product Requirements Document (PRD)"
        ],
        "Finance & Operations": [
            "Redattore di Comunicazioni agli Stakeholder (Finanziarie)",
            "Generatore di Policy Aziendali (Bozza Strategica)"
        ],
        "Human Resources & Talent Management": [
            "Architetto di Programmi di Onboarding",
            "Disegnatore di Piani di Sviluppo Carriera",
            "Pianificatore di Campagne di Employer Branding",
            "Strutturatore di Sondaggi sul Coinvolgimento dei Dipendenti"
        ]
    };

    const [selectedProfile, setSelectedProfile] = useState("Sviluppatore di Business Model Canvas");
    const { isSignedIn, getToken, signOut } = useAuth();

    const handleStrategyGeneration = async () => {
        if (!isSignedIn) {
            alert("Devi essere autenticato per usare lo Strategist.");
            router.push('/login');
            return;
        }
        if (!inputText.trim()) {
            alert('Per favore, inserisci un input per generare la strategia.');
            return;
        }
        setIsLoading(true);
        setOutputText('Generazione della strategia in corso...');

        try {
            const token = await getToken();
            if (!token) throw new Error("Impossibile ottenere il token.");

            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/strategist`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
                body: JSON.stringify({ text: inputText, profile_name: selectedProfile }),
            });

            const data = await response.json();

            if (!response.ok) {
                if (response.status === 401) {
                    signOut(() => router.push('/login'));
                    return;
                }
                if (response.status === 403) {
                    router.push('/pricing');
                    alert(data.detail || "Questa funzionalità non è inclusa nel tuo piano. Esegui l'upgrade per continuare.");
                    return;
                }
                throw new Error(data.detail || 'Si è verificato un errore.');
            }
            
            setOutputText(data.strategy_text);
            await fetchUserStatus();
            
        } catch (error) {
            if (error instanceof Error) setOutputText(`Errore: ${error.message}`);
            else setOutputText('Si è verificato un errore sconosciuto.');
        } finally {
            setIsLoading(false);
        }
    };

    const handleCopy = () => {
        if (!outputText || isLoading) return;
        navigator.clipboard.writeText(outputText).then(() => {
            setCopyButtonText('Copiato!');
            setTimeout(() => setCopyButtonText('Copia'), 2000);
        });
    };

    const handleClear = () => {
        setInputText('');
        setOutputText('');
    };

    // Gate per utenti non Pro
    if (!strategistAccess) {
        return (
            <main className="flex flex-col items-center justify-center min-h-screen bg-gray-900 px-4 md:px-8 text-white">
                <div className="text-center p-8 bg-gray-800 rounded-lg shadow-lg border border-gray-700">
                    <h2 className="text-3xl font-bold text-purple-400 mb-4">Accesso Esclusivo Piano Pro</h2>
                    <p className="text-lg text-gray-300 mb-6">
                        Lo <strong>Strategist AI</strong> è la nostra funzionalità più avanzata, disponibile solo per gli utenti Pro.
                    </p>
                    <button
                        onClick={() => router.push('/pricing')}
                        className="rounded-xl bg-blue-600 px-8 py-3 text-lg font-bold text-white shadow-lg hover:bg-blue-500 transition-all"
                    >
                        Scopri il Piano Pro
                    </button>
                </div>
            </main>
        );
    }

    return (
        <main className="flex flex-col items-center bg-gray-900 p-8 text-white">
            <div className="w-full max-w-4xl">
                <header className="mb-8 text-center">
                    <h1 className="text-4xl font-extrabold text-purple-400">Strategist AI</h1>
                    <p className="mt-2 text-lg text-gray-400 font-semibold">
                        Dall'idea all'azione. Genera framework, piani e bozze strategiche.
                    </p>
                </header>

                <div className="mb-6">
                    <label htmlFor="aiProfile" className="block text-sm font-medium text-gray-300 mb-2">
                        Seleziona Profilo Strategico
                    </label>
                    <select
                        id="aiProfile"
                        value={selectedProfile}
                        onChange={(e) => setSelectedProfile(e.target.value)}
                        disabled={isLoading}
                        className="w-full rounded-lg border border-gray-700 bg-gray-900 p-3 text-gray-200 shadow-sm focus:border-purple-500 focus:ring-purple-500"
                    >
                        {Object.entries(strategistProfileGroups).map(([groupName, profiles]) => (
                            <optgroup key={groupName} label={groupName}>
                                {profiles.map((profile) => (
                                    <option key={profile} value={profile}>{profile}</option>
                                ))}
                            </optgroup>
                        ))}
                    </select>
                </div>
            
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <div className="flex justify-between items-center mb-2">
                            <label htmlFor="inputText" className="block text-sm font-medium text-gray-300">
                                Inserisci la tua idea o il contesto
                            </label>
                            <button
                                onClick={handleClear}
                                disabled={!inputText || isLoading}
                                className="rounded-lg bg-gray-700 px-4 py-2 text-sm font-semibold text-white shadow-md hover:bg-gray-600"
                            >
                                Pulisci
                            </button>
                        </div>
                        <textarea
                            id="inputText"
                            rows={15}
                            className="w-full rounded-lg border border-gray-700 bg-gray-900 p-3 text-gray-200 shadow-sm focus:border-purple-500 focus:ring-purple-500 resize-none"
                            placeholder="Es: Un'idea per un SaaS B2B che analizza il sentiment dei clienti..."
                            value={inputText}
                            onChange={(e) => setInputText(e.target.value)}
                            disabled={isLoading}
                        />
                    </div>

                    <div>
                        <div className="flex justify-between items-center mb-2">
                            <label className="block text-sm font-medium text-gray-300">
                                Output Strategico
                            </label>
                            <button
                                onClick={handleCopy}
                                disabled={!outputText || isLoading}
                                className="rounded-lg bg-gray-700 px-4 py-2 text-sm font-semibold text-white shadow-md hover:bg-gray-600"
                            >
                                {copyButtonText}
                            </button>
                        </div>
                        <div 
                            style={{ height: '385px' }}
                            className="w-full rounded-lg border border-gray-700 bg-gray-800 p-4 text-gray-300 shadow-inner overflow-y-auto prose prose-invert prose-p:text-gray-300 prose-strong:text-white prose-headings:text-purple-400"
                        >
                            <ReactMarkdown>
                                {outputText || "Il piano strategico apparirà qui..."}
                            </ReactMarkdown>
                        </div>
                    </div>
                </div>

                <div className="mt-6 flex justify-center">
                    <button
                        onClick={handleStrategyGeneration}
                        disabled={isLoading || !inputText}
                        className="rounded-xl bg-purple-600 px-10 py-4 text-xl font-bold text-white shadow-lg hover:bg-purple-500 transition-all disabled:bg-gray-700 disabled:cursor-not-allowed"
                    >
                        {isLoading ? 'Elaborazione...' : 'Genera Strategia'}
                    </button>
                </div>
            </div>
        </main>
    );
}