import { afterEach, describe, expect, it, vi } from "vitest";

import { apiFetch } from "./useAuth.js";

describe("apiFetch refresh coordination", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("coalesces concurrent 401 responses into one refresh-token rotation", async () => {
    const attempts = new Map();
    let refreshCalls = 0;
    const fetchMock = vi.fn(async (url) => {
      if (url.endsWith("/auth/refresh")) {
        refreshCalls += 1;
        return { ok: true, status: 200 };
      }

      const count = attempts.get(url) ?? 0;
      attempts.set(url, count + 1);
      return count === 0
        ? { ok: false, status: 401 }
        : { ok: true, status: 200 };
    });
    vi.stubGlobal("fetch", fetchMock);

    const responses = await Promise.all([
      apiFetch("/history"),
      apiFetch("/fixtures/upcoming"),
    ]);

    expect(responses.every((response) => response.ok)).toBe(true);
    expect(refreshCalls).toBe(1);
    expect(fetchMock).toHaveBeenCalledTimes(5);
  });
});
