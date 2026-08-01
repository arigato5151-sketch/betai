import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import AnalysisForm from "./AnalysisForm.jsx";

const formData = {
  home_team: "Fenerbahçe",
  away_team: "Galatasaray",
  league_id: null,
  odd: 2.3,
  home_stats: { form: 93, attack: 88, defense: 85, xg: 2.15 },
  away_stats: { form: 73, attack: 82, defense: 75, xg: 1.9 },
};

const leagues = [
  { id: 2, name: "UEFA Champions League" },
  { id: 3, name: "UEFA Europa League" },
  { id: 848, name: "UEFA Europa Conference League" },
  { id: 39, name: "Premier League" },
];

describe("AnalysisForm lig seçimi", () => {
  it("UEFA liglerini Türkçe, diğer ligleri canonical adlarıyla gösterir", () => {
    const onChange = vi.fn();
    render(
      <AnalysisForm
        formData={formData}
        leagues={leagues}
        loading={false}
        onChange={onChange}
        onSubmit={vi.fn()}
      />,
    );

    const select = screen.getByRole("combobox", { name: "Lig" });
    expect(
      screen.getByRole("option", { name: "UEFA Şampiyonlar Ligi" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("option", { name: "UEFA Avrupa Ligi" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("option", { name: "UEFA Konferans Ligi" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("option", { name: "Premier League" }),
    ).toBeInTheDocument();

    fireEvent.change(select, { target: { value: "848" } });

    expect(onChange).toHaveBeenCalledWith({
      ...formData,
      league_id: 848,
    });
    expect(typeof onChange.mock.calls[0][0].league_id).toBe("number");
  });

  it("yükleme durumunu erişilebilir biçimde bildirir", () => {
    render(
      <AnalysisForm
        formData={formData}
        leaguesLoading
        loading={false}
        onChange={vi.fn()}
        onSubmit={vi.fn()}
      />,
    );

    const select = screen.getByRole("combobox", { name: "Lig" });
    expect(select).toBeDisabled();
    expect(select).toHaveAttribute("aria-busy", "true");
    expect(screen.getByRole("status")).toHaveTextContent(
      "Desteklenen ligler yükleniyor",
    );
  });

  it("lig listesi yoksa hata ve manuel devam fallback'ini gösterir", () => {
    render(
      <AnalysisForm
        formData={formData}
        leaguesError="Desteklenen ligler alınamadı. Lig seçmeden manuel analize devam edebilirsiniz."
        loading={false}
        onChange={vi.fn()}
        onSubmit={vi.fn()}
      />,
    );

    expect(screen.getByRole("combobox", { name: "Lig" })).toBeDisabled();
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Lig seçmeden manuel analize devam edebilirsiniz",
    );
    expect(
      screen.getByRole("button", { name: "Tahmin Oluştur" }),
    ).toBeEnabled();
  });

  it("fikstür verisi yüklenirken analizi geçici olarak devre dışı bırakır", () => {
    render(
      <AnalysisForm
        fixtureLoading
        formData={formData}
        leagues={leagues}
        loading={false}
        onChange={vi.fn()}
        onSubmit={vi.fn()}
      />,
    );

    expect(
      screen.getByRole("button", { name: "Maç verileri yükleniyor…" }),
    ).toBeDisabled();
  });

  it("manuel oran değişikliğinde otomatik odds snapshot çiftini temizler", () => {
    const onChange = vi.fn();
    render(
      <AnalysisForm
        formData={{
          ...formData,
          market_1x2: {
            raw_odds: { HOME_WIN: 2.3, DRAW: 3.2, AWAY_WIN: 3.4 },
          },
          opening_odds_1x2: {
            HOME_WIN: 2.5,
            DRAW: 3.1,
            AWAY_WIN: 3.2,
          },
          current_odds_1x2: {
            HOME_WIN: 2.3,
            DRAW: 3.2,
            AWAY_WIN: 3.4,
          },
          opening_odds_at: "2030-07-29T09:00:00+00:00",
          current_odds_at: "2030-07-30T09:00:00+00:00",
        }}
        leagues={leagues}
        loading={false}
        onChange={onChange}
        onSubmit={vi.fn()}
      />,
    );

    fireEvent.change(screen.getByLabelText("Bahis oranı"), {
      target: { value: "2.2" },
    });

    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({
        opening_odds_1x2: null,
        current_odds_1x2: null,
        opening_odds_at: null,
        current_odds_at: null,
      }),
    );
  });

  it("temel istatistikleri gösterir ancak ayrıntılı model girdilerini ekrana basmaz", () => {
    render(
      <AnalysisForm
        formData={{ ...formData, feature_overrides: {} }}
        leagues={leagues}
        loading={false}
        onChange={vi.fn()}
        onSubmit={vi.fn()}
      />,
    );

    expect(screen.getAllByLabelText("Hücum Gücü")).toHaveLength(2);
    expect(screen.getAllByLabelText("Savunma Gücü")).toHaveLength(2);
    expect(screen.getAllByLabelText("Gol Beklentisi (xG)")).toHaveLength(2);
    expect(
      screen.queryByText("Hesaplanan Tüm Model Girdileri"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByLabelText("Yorgunluk ve Seyahat Endeksi"),
    ).not.toBeInTheDocument();
  });
});
