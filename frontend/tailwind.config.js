/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'cyber-black': '#0a0a0a',
        'cyber-gray': '#1a1a1a',
        'neon-blue': '#00f3ff',
        'neon-green': '#0aff00',
        'neon-red': '#ff003c',
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'monospace'],
        sans: ['Inter', 'sans-serif'],
      },
      boxShadow: {
        'neon-blue': '0 0 10px #00f3ff, 0 0 20px #00f3ff',
        'neon-green': '0 0 10px #0aff00, 0 0 20px #0aff00',
        'neon-red': '0 0 10px #ff003c, 0 0 20px #ff003c',
      }
    },
  },
  plugins: [],
}
