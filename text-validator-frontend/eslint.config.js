const nextEslintConfig = require('next/eslint');

/** @type {import('eslint').Linter.Config} */
const config = {
  ...nextEslintConfig,
  rules: {
    ...nextEslintConfig.rules,
    // Qui puoi aggiungere o modificare regole se necessario
  },
};

module.exports = config;