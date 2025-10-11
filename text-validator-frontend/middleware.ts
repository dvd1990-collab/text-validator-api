// middleware.ts
import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server';

// Definisci le rotte pubbliche che non richiedono autenticazione.
// Assicurati che le pagine di login/signup e l'endpoint del webhook siano pubblici.
const isPublicRoute = createRouteMatcher([
  '/sign-in(.*)',
  '/sign-up(.*)',
  '/', // La tua homepage se vuoi che sia accessibile prima del login (ma poi Clerk reindirizzerà se l'app è protetta)
  '/api/webhook/clerk' // Importante: l'endpoint del webhook deve essere pubblico
]);

export default clerkMiddleware((auth, request) => { 
  // Se la richiesta NON è per una rotta pubblica, proteggila.
  if (!isPublicRoute(request)) {
    auth.protect(); // <--- CORREZIONE QUI: Chiamiamo protect SENZA argomenti
  }
});

// Configurazione standard per il matcher di Next.js
export const config = {
  matcher: ['/((?!.*\\..*|_next).*)', '/', '/(api|trpc)(.*)'],
};