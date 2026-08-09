import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import BankrollChart from "./BankrollChart.jsx";

vi.mock("react-chartjs-2", () => ({
  Line: () => <div data-testid="bankroll-chart-renderer" />,
}));

describe("BankrollChart araştırma güvenliği", () => {
  it("simülasyonu finansal öneri olarak sunmaz", () => {
    const onRun = vi.fn();
    render(
      <BankrollChart
        backtest={null}
        bankrollSeries={{ labels: [], values: [], change: 0 }}
        error=""
        loading={false}
        onRun={onRun}
      />,
    );

    expect(screen.getByText(/finansal öneri değildir/i)).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: "Araştırma Testini Çalıştır" }),
    );
    expect(onRun).toHaveBeenCalledOnce();
  });

  it("sıfır bahis için anlamsız performans metrikleri göstermez", () => {
    render(
      <BankrollChart
        backtest={{
          total_bets: 0,
          skipped_reasons: { missing_closing_odds: 6 },
        }}
        bankrollSeries={{ labels: ["Başlangıç"], values: [1000], change: 0 }}
        error=""
        loading={false}
        onRun={vi.fn()}
      />,
    );

    expect(
      screen.getByText("Değerlendirilebilir bahis kaydı yok"),
    ).toBeInTheDocument();
    expect(screen.getByText(/Kapanış oranı eksik: 6/)).toBeInTheDocument();
    expect(screen.queryByText("Yatırım Getirisi (ROI)")).not.toBeInTheDocument();
    expect(screen.queryByTestId("bankroll-chart-renderer")).not.toBeInTheDocument();
  });
});
