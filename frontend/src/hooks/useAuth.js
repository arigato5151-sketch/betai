import { useEffect, useState } from "react";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api"
).replace(/\/$/, "");
const CSRF_COOKIE_NAME = import.meta.env.VITE_CSRF_COOKIE_NAME || "bet_ai_csrf";
const CSRF_HEADER_NAME = import.meta.env.VITE_CSRF_HEADER_NAME || "X-CSRF-Token";

const readCookie = (name) =>
  document.cookie
    .split(";")
    .map((part) => part.trim())
    .find((part) => part.startsWith(`${name}=`))
    ?.slice(name.length + 1) || "";

const addCsrfHeader = (headers, method = "GET") => {
  if (!["POST", "PUT", "PATCH", "DELETE"].includes(method.toUpperCase())) return;
  const token = readCookie(CSRF_COOKIE_NAME);
  if (token) headers.set(CSRF_HEADER_NAME, decodeURIComponent(token));
};

const refreshAccessToken = async () => {
  const headers = new Headers({ "Content-Type": "application/json" });
  addCsrfHeader(headers, "POST");
  const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
    method: "POST",
    credentials: "include",
    headers,
  });
  return response.ok;
};

export const apiFetch = async (path, options = {}, allowRefresh = true) => {
  const headers = new Headers(options.headers || {});
  addCsrfHeader(headers, options.method || "GET");
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
    credentials: "include",
  });
  if (response.status !== 401 || !allowRefresh) return response;

  const renewed = await refreshAccessToken();
  if (!renewed) {
    window.dispatchEvent(new Event("bet-ai:unauthorized"));
    return response;
  }

  // Refresh rotates the CSRF cookie together with the token pair.
  addCsrfHeader(headers, options.method || "GET");
  return fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
    credentials: "include",
  });
};

export function useAuth() {
  const [authenticated, setAuthenticated] = useState(null);
  const [sessionUser, setSessionUser] = useState(null);
  const [credentials, setCredentials] = useState({ username: "", email: "", password: "" });
  const [registerMode, setRegisterMode] = useState(false);
  const [loginError, setLoginError] = useState("");
  const [loginLoading, setLoginLoading] = useState(false);

  useEffect(() => {
    apiFetch("/auth/session", {}, false)
      .then(async (response) => {
        if (!response.ok) {
          setAuthenticated(false);
          setSessionUser(null);
          return;
        }
        const session = await response.json();
        setSessionUser(session.user);
        setAuthenticated(true);
      })
      .catch(() => {
        setAuthenticated(false);
        setSessionUser(null);
      });
  }, []);

  useEffect(() => {
    const handleUnauthorized = () => {
      setAuthenticated(false);
      setSessionUser(null);
    };
    window.addEventListener("bet-ai:unauthorized", handleUnauthorized);
    return () => window.removeEventListener("bet-ai:unauthorized", handleUnauthorized);
  }, []);

  const login = async (event) => {
    event.preventDefault();
    setLoginLoading(true);
    setLoginError("");
    try {
      const endpoint = registerMode ? "/auth/register" : "/auth/login";
      const payload = registerMode ? credentials : {
        username: credentials.username,
        password: credentials.password,
      };
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        throw new Error(errorBody.detail || "Oturum işlemi başarısız.");
      }
      const session = await response.json();
      setSessionUser(session.user);
      setAuthenticated(true);
      setCredentials({ username: "", email: "", password: "" });
      setRegisterMode(false);
    } catch (error) {
      setLoginError(error.message || "Oturum açılamadı.");
    } finally {
      setLoginLoading(false);
    }
  };

  const logout = async () => {
    await apiFetch("/auth/logout", { method: "POST" }, false);
    setAuthenticated(false);
    setSessionUser(null);
  };

  const toggleRegisterMode = () => {
    setRegisterMode((current) => !current);
    setLoginError("");
  };

  return {
    authenticated,
    credentials,
    login,
    loginError,
    loginLoading,
    logout,
    registerMode,
    sessionUser,
    setCredentials,
    toggleRegisterMode,
  };
}
