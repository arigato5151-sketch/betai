import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import ModelStatusCard from "./ModelStatusCard.jsx";

const status = {
  ready: true,
  model_name: "Random Forest",
  artifact_version: "model-v1",
  metrics: {
    samples: 100,
    brier_score: 0.2,
    baseline_brier_score: 0.25,
    calibration_error: 0.04,
    accuracy: 0.61,
    league_count: 4,
  },
  runtime: { inference_success: 12, inference_failure: 2 },
  rollback_available: true,
  training_data: { historical_fixtures: 500, labeled_predictions: 80 },
  monitoring: {
    status: "insufficient_data",
    samples: 18,
    required_samples: 60,
    recent_brier: null,
    brier_delta: null,
    brier_delta_lower_bound: null,
    confidence: 0.95,
  },
  live_evaluation: {
    status: "insufficient_data",
    verified_samples: 18,
    required_samples: 60,
    claims_enabled: false,
    artifact_version: "model-v1",
  },
};

describe("ModelStatusCard", () => {
  it("artifact ve drift güven bilgilerini operasyon dilinde gösterir", () => {
    render(
      <ModelStatusCard
        status={status}
        error=""
        loading={false}
        onRefresh={vi.fn()}
      />,
    );

    expect(screen.getByText("model-v1")).toBeInTheDocument();
    expect(screen.getByText("Yetersiz Doğrulanmış Örnek")).toBeInTheDocument();
    expect(screen.getByText("18/60")).toBeInTheDocument();
    expect(screen.getByText("%95")).toBeInTheDocument();
    expect(screen.getByText("Hazır")).toBeInTheDocument();
    expect(screen.getByText("Test Doğruluğu")).toBeInTheDocument();
    expect(screen.getByText("Ölçüm kapalı")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(
      "aktif artifact için 18/60 doğrulanmış tahmin",
    );
  });

  it("yenileme ve hata durumlarını erişilebilir duyurur", () => {
    const onRefresh = vi.fn();
    const { rerender } = render(
      <ModelStatusCard
        status={null}
        error=""
        loading
        onRefresh={onRefresh}
      />,
    );

    expect(screen.getByRole("status")).toHaveTextContent(
      "Model operasyon durumu yükleniyor.",
    );
    fireEvent.click(screen.getByRole("button", { name: "Yenileniyor..." }));
    expect(onRefresh).not.toHaveBeenCalled();

    rerender(
      <ModelStatusCard
        status={null}
        error="Model durumu alınamadı."
        loading={false}
        onRefresh={onRefresh}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Model durumu alınamadı.");
  });
});
