import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import UpcomingFixturesContainer, {
  normalizeUpcomingFixtures,
} from "./UpcomingFixturesContainer.jsx";

const unsortedFixtures = [
  {
    fixture_id: 30,
    league_id: 2,
    league: "UEFA Champions League",
    home_team: "Geç Takım",
    away_team: "Rakip C",
    kickoff: "2030-07-31T21:00:00+03:00",
    is_demo: false,
    source: "openligadb",
    sources: ["openligadb"],
  },
  {
    fixture_id: 10,
    league_id: 848,
    league: "UEFA Europa Conference League",
    home_team: "Erken Takım",
    away_team: "Rakip A",
    kickoff: "2030-07-30T18:00:00+03:00",
    is_demo: false,
  },
  {
    fixture_id: 20,
    league_id: 3,
    league: "UEFA Europa League",
    home_team: "Orta Takım",
    away_team: "Rakip B",
    kickoff: "2030-07-30T20:00:00+03:00",
    is_demo: true,
  },
];

describe("UpcomingFixturesContainer", () => {
  it("fikstürü yedi günlük endpoint'ten alır ve kronolojik gösterir", async () => {
    const request = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => unsortedFixtures,
    });
    const onSelectFixture = vi.fn();

    render(
      <UpcomingFixturesContainer
        onSelectFixture={onSelectFixture}
        request={request}
        selectedFixtureId={10}
      />,
    );

    await screen.findByText("Erken Takım");
    expect(request).toHaveBeenCalledWith(
      "/fixtures/upcoming?days=7&limit=100",
    );
    const cards = screen.getAllByRole("article");
    expect(cards.map((card) => card.textContent)).toEqual([
      expect.stringContaining("Erken Takım"),
      expect.stringContaining("Orta Takım"),
      expect.stringContaining("Geç Takım"),
    ]);
    expect(screen.getByText("UEFA Konferans Ligi")).toBeInTheDocument();
    expect(screen.getByText("Demo veri")).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "OpenLigaDB (ODbL)" }),
    ).toHaveAttribute("href", "https://www.openligadb.de/");
    expect(screen.getByText("Analize seçildi")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", {
        name: "Erken Takım – Rakip A maçını analiz formuna taşı",
      }),
    );
    expect(onSelectFixture).toHaveBeenCalledWith(
      expect.objectContaining({ fixture_id: 10 }),
    );
  });

  it("yenileme düğmesiyle fikstürü yeniden ister", async () => {
    const request = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => unsortedFixtures,
    });
    render(<UpcomingFixturesContainer request={request} />);
    await screen.findByText("Erken Takım");

    fireEvent.click(screen.getByRole("button", { name: "Fikstürü Yenile" }));

    await waitFor(() => expect(request).toHaveBeenCalledTimes(2));
  });

  it("istek başarısızlığını erişilebilir biçimde gösterir", async () => {
    const request = vi.fn().mockRejectedValue(new Error("Bağlantı kurulamadı."));
    render(<UpcomingFixturesContainer request={request} />);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Bağlantı kurulamadı.",
    );
  });
});

describe("normalizeUpcomingFixtures", () => {
  it("geçersiz satırları eler ve eşit tarihte fixture ID ile kararlı sıralar", () => {
    const result = normalizeUpcomingFixtures([
      unsortedFixtures[0],
      { ...unsortedFixtures[1], fixture_id: 11 },
      unsortedFixtures[1],
      { fixture_id: 999, kickoff: "invalid" },
    ]);

    expect(result.map((fixture) => fixture.fixture_id)).toEqual([10, 11, 30]);
  });
});
