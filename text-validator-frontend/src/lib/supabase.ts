import { createClient } from '@supabase/supabase-js';

// Recupera le variabili d'ambiente Supabase
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

// Assicurati che le chiavi siano presenti
if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error(
    'Missing Supabase URL or Anon Key. ' +
      'Ensure NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY ' +
      'are set as environment variables.'
  );
}

// Inizializza il client Supabase
export const supabase = createClient(supabaseUrl, supabaseAnonKey);