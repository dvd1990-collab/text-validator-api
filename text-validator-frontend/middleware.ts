// middleware.ts
import { authMiddleware } from "@clerk/nextjs/server";

export default authMiddleware({
  // Rotte che sono accessibili a chiunque (non protette)
  // '/login', '/sign-up', '/' (se vuoi che la homepage sia pubblica prima del login)
  publicRoutes: ["/", "/login", "/sign-up"],

  // Ignora le rotte che Next.js e Clerk gestiscono internamente
  // Questo pattern è cruciale per prevenire i 404 sui file interni di Clerk e Next.js
  ignoredRoutes: ['/((?!api|trpc|_next|_static|favicon.ico).*)'], // Mantieni questa se usi una versione più vecchia di Clerk che la richiede
  // Oppure puoi provare una versione più semplice se quella sopra non funziona:
  // ignoredRoutes: ["/((?!.+\\.[\\w]+$|_next).*)"], 
});

// La configurazione del matcher è ESSENZIALE per Next.js per sapere cosa proteggere
// e cosa lasciare al middleware di Clerk.
export const config = {
  matcher: [
    // Rotte da proteggere (tutto tranne i file statici e le rotte interne di Next.js)
    '/((?!.+\\.[\\w]+$|_next).*)',
    // Includi le rotte API che vuoi proteggere (se ne hai, es. /api/auth)
    // '/(api|trpc)(.*)', 
  ],
};