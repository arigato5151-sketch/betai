import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(() => {
  const proxyTarget = process.env.VITE_PROXY_TARGET || "http://127.0.0.1:8000";
  const apiProxy = {
    "/api": {
      target: proxyTarget,
      changeOrigin: true,
    },
  };

  return {
    plugins: [react()],
    server: {
      proxy: apiProxy,
    },
    preview: {
      proxy: apiProxy,
    },
    test: {
      environment: "jsdom",
      setupFiles: "./src/testSetup.js",
      include: ["src/**/*.component.test.{js,jsx}"],
      clearMocks: true,
    },
  };
});
