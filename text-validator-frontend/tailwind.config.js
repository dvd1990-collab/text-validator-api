// tailwind.config.js

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}", // <-- QUESTA RIGA ORA DICE A TAILWIND DI GUARDARE NEI COMPONENTI
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      // Qui puoi aggiungere estensioni al tema di Tailwind in futuro
      // es. colori personalizzati, font, etc.
    },
  },
  plugins: [],
};