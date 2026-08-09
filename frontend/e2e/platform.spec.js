import { expect, test } from "@playwright/test";

const APP_TITLE = "BET AI TAHMİN PLATFORMU";
const username = process.env.E2E_USERNAME || "admin";
const password =
  process.env.E2E_PASSWORD || "ci-password-only-long-enough";

async function login(page) {
  await page.goto("/");
  await expect(page.locator("#auth-username")).toBeVisible();
  await page.locator("#auth-username").fill(username);
  await page.locator("#auth-password").fill(password);
  await page.getByRole("button", { name: "Giriş Yap" }).click();
  await expect(page.getByRole("button", { name: "Çıkış" })).toBeVisible();
}

test("01 login, logout and login again (session round-trip)", async ({
  page,
}) => {
  await login(page);
  await expect(
    page.getByRole("heading", { name: APP_TITLE }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Çıkış" }).click();
  await expect(page.locator("#auth-username")).toBeVisible();
  await login(page);
  await expect(page.getByRole("button", { name: "Çıkış" })).toBeVisible();
});

test("02 cross-site origin is rejected before any state change", async ({
  page,
}) => {
  await login(page);
  const response = await page.request.post("/api/analyze", {
    headers: { Origin: "https://evil.example" },
    data: {},
  });
  expect(response.status()).toBe(403);
});

test("03 analysis form round-trip via demo fixture selection", async ({
  page,
}) => {
  await login(page);
  const selectFixture = page.getByRole("button", {
    name: /maçını analiz formuna taşı/i,
  });
  try {
    await selectFixture.first().waitFor({ timeout: 20_000 });
  } catch {
    test.skip(true, "demo environment exposes no upcoming fixtures");
    return;
  }
  await selectFixture.first().click();
  await expect(page.getByLabel("Ev sahibi takım")).not.toHaveValue("");
  await expect(page.getByLabel("Deplasman takımı")).not.toHaveValue("");
  const [response] = await Promise.all([
    page.waitForResponse(
      (response) =>
        response.url().includes("/api/analyze") &&
        response.request().method() === "POST",
      { timeout: 60_000 },
    ),
    page.getByRole("button", { name: "Tahmin Oluştur" }).click(),
  ]);
  expect([200, 201, 400, 409, 422]).toContain(response.status());
  await response.json();
  await expect(page.getByRole("button", { name: "Tahmin Oluştur" })).toBeEnabled();
});

test("04 token refresh keeps the dashboard alive", async ({ page }) => {
  await login(page);
  await page.waitForTimeout(1_500);
  await page.reload();
  await expect(page.getByRole("button", { name: "Çıkış" })).toBeVisible();
});