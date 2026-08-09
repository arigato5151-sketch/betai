import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { useRequestGate } from "./useRequestGate.js";

describe("useRequestGate stale-state ayrımı", () => {
  it("daha eski bir yanıt gecikerek geldiğinde bunu stale işaretler ve uygulamaz", () => {
    const { result } = renderHook(() => useRequestGate());

    let firstApply = false;
    let secondApply = false;
    let first;
    let second;
    act(() => {
      first = result.current.begin();
      second = result.current.begin();
    });

    act(() => {
      const applied = result.current.settle(first, () => {
        firstApply = true;
      });
      expect(applied).toBe(false);
    });
    expect(firstApply).toBe(false);
    expect(result.current.stale).toBe(true);

    act(() => {
      const applied = result.current.settle(second, () => {
        secondApply = true;
      });
      expect(applied).toBe(true);
    });
    expect(secondApply).toBe(true);
    expect(result.current.stale).toBe(false);
  });

  it("tamamen bitmiş bir istek sonrası gelen eski yanıt stale işareti bırakmaz", () => {
    const { result } = renderHook(() => useRequestGate());

    let first;
    act(() => {
      first = result.current.begin();
      result.current.settle(first, () => {});
    });
    expect(result.current.stale).toBe(false);

    let second;
    act(() => {
      second = result.current.begin();
    });
    act(() => {
      const applied = result.current.settle(first, () => {});
      expect(applied).toBe(false);
    });
    expect(result.current.stale).toBe(false);

    act(() => {
      result.current.settle(second, () => {});
    });
    expect(result.current.stale).toBe(false);
  });
});