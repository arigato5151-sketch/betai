import { useRef, useState } from "react";

export function useRequestGate() {
  const versionRef = useRef(0);
  const inflightRef = useRef(0);
  const [stale, setStale] = useState(false);

  const begin = () => {
    const mine = ++versionRef.current;
    inflightRef.current = mine;
    setStale(false);
    return mine;
  };

  const settle = (mine, apply) => {
    if (mine !== versionRef.current) {
      if (inflightRef.current !== 0) setStale(true);
      return false;
    }
    inflightRef.current = 0;
    apply();
    setStale(false);
    return true;
  };

  return { stale, begin, settle };
}