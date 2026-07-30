import { useEffect, useState } from "react";

import DataQualityCard from "../components/DataQualityCard.jsx";
import ModelStatusCard from "../components/ModelStatusCard.jsx";

function OperationsContainer({ actions, request }) {
  const [dataQuality, setDataQuality] = useState(null);
  const [dataQualityLoading, setDataQualityLoading] = useState(false);
  const [dataQualityError, setDataQualityError] = useState("");
  const [modelStatus, setModelStatus] = useState(null);
  const [modelStatusLoading, setModelStatusLoading] = useState(false);
  const [modelStatusError, setModelStatusError] = useState("");

  const fetchDataQuality = async () => {
    if (!actions.readAudit) return;
    setDataQualityLoading(true);
    setDataQualityError("");
    try {
      const response = await request("/operations/data-quality");
      if (!response.ok) throw new Error("Veri kalitesi durumu alınamadı.");
      setDataQuality(await response.json());
    } catch (error) {
      setDataQualityError(
        error.message || "Veri kalitesi durumu alınamadı.",
      );
    } finally {
      setDataQualityLoading(false);
    }
  };

  const fetchModelStatus = async () => {
    if (!actions.readHistory) return;
    setModelStatusLoading(true);
    setModelStatusError("");
    try {
      const response = await request("/ml/status");
      if (!response.ok) {
        throw new Error("Makine öğrenmesi modeli durumu alınamadı.");
      }
      setModelStatus(await response.json());
    } catch (error) {
      setModelStatusError(
        error.message || "Makine öğrenmesi modeli durumu alınamadı.",
      );
    } finally {
      setModelStatusLoading(false);
    }
  };

  useEffect(() => {
    if (actions.readAudit) fetchDataQuality();
  }, [actions.readAudit]);

  useEffect(() => {
    if (actions.readHistory) fetchModelStatus();
  }, [actions.readHistory]);

  return (
    <>
      {actions.readAudit && (
        <DataQualityCard
          data={dataQuality}
          error={dataQualityError}
          loading={dataQualityLoading}
          onRefresh={fetchDataQuality}
        />
      )}

      {actions.readHistory && (
        <ModelStatusCard
          status={modelStatus}
          error={modelStatusError}
          loading={modelStatusLoading}
          onRefresh={fetchModelStatus}
        />
      )}
    </>
  );
}

export default OperationsContainer;
