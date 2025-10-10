// src/app/sign-up/page.tsx
import { SignUp } from "@clerk/nextjs";

// Legge la variabile d'ambiente
const allowRegistration = process.env.NEXT_PUBLIC_ALLOW_REGISTRATION === 'true';

export default function SignUpPage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-900 p-8 text-white">
      <div className="w-full max-w-md">
        <h1 className="text-3xl font-bold text-center text-blue-400 mb-6">
          Registrati a Text Validator
        </h1>
        {/* Logica condizionale per mostrare il form o il messaggio */}
        {allowRegistration ? (
          <SignUp
            signInUrl="/login"
            afterSignUpUrl="/"
            routing="hash"
          />
        ) : (
          <div className="text-center p-6 rounded-md bg-gray-700 border border-gray-600 text-gray-300">
            <p className="text-lg font-semibold mb-2">Registrazioni Temporaneamente Chiuse</p>
            <p>Stiamo effettuando importanti aggiornamenti. Riprova più tardi o contatta il supporto.</p>
            <a href="/login" className="mt-4 inline-block text-blue-400 hover:text-blue-300">Torna al Login</a>
          </div>
        )}
      </div>
    </main>
  );
}