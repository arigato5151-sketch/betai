import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: "./src/testSetup.js",
    include: ["src/**/*.component.test.{js,jsx}"],
    clearMocks: true,
  },
});
