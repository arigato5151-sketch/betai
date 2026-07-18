import js from "@eslint/js";
import react from "eslint-plugin-react";


export default [
  {
    ignores: ["dist/**"],
  },
  {
    files: ["src/**/*.{js,jsx}"],
    ...js.configs.recommended,
    plugins: { react },
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
      globals: {
        alert: "readonly",
        Event: "readonly",
        fetch: "readonly",
        Headers: "readonly",
        URLSearchParams: "readonly",
        window: "readonly",
      },
    },
    rules: {
      "no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
      "react/jsx-uses-vars": "error",
    },
  },
];
