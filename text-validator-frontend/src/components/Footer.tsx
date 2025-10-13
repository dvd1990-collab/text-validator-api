// src/components/Footer.tsx
import Link from 'next/link';

export default function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="w-full bg-gray-800 border-t border-gray-700 mt-20">
      <div className="container mx-auto px-8 py-8 flex flex-col md:flex-row justify-between items-center text-center md:text-left">
        <p className="text-gray-400 text-sm mb-4 md:mb-0">
          © {currentYear} Text Validator AI. Tutti i diritti riservati.
        </p>
        <div className="flex space-x-6">
          <Link href="/privacy-policy" className="text-gray-300 hover:text-blue-400 transition-colors text-sm">
            Privacy Policy
          </Link>
          <Link href="/terms-of-use" className="text-gray-300 hover:text-blue-400 transition-colors text-sm">
            Terms of Use
          </Link>
          {/* Sostituisci con la tua vera email di supporto quando sarà pronta */}
          <a href="mailto:supporto@textvalidator.ai" className="text-gray-300 hover:text-blue-400 transition-colors text-sm">
            Contact Us
          </a>
        </div>
      </div>
    </footer>
  );
}