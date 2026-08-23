interface Env {
  DB: D1Database;
  API_FOOTBALL_KEY: string;
  CLIENT_HMAC_SECRET: string;
  DAILY_API_BUDGET: string;
  CLOSING_RESERVE: string;
  CLOSING_FIRST_WINDOW_MINUTES: string;
  CLOSING_RETRY_WINDOW_MINUTES: string;
}

interface TrackFixtureRequest {
  id: string;
  api_fixture_id?: number;
  league_id: number;
  home_team: string;
  away_team: string;
  kickoff: string;
  preferred_bookmaker?: string;
  capture_entry?: boolean;
}

interface TrackedFixture {
  id: string;
  api_fixture_id: number | null;
  league_id: number;
  home_team: string;
  away_team: string;
  kickoff: string;
  preferred_bookmaker: string | null;
  closing_attempts: number;
}

interface OddsSelection {
  home: number;
  draw: number;
  away: number;
  bookmaker: string;
  providerUpdatedAt: string | null;
  details: Record<string, unknown>;
}

interface ApiResponse<T> {
  response?: T[];
  errors?: Record<string, unknown> | unknown[];
}

const JSON_HEADERS = {
  "content-type": "application/json; charset=utf-8",
  "cache-control": "no-store",
};
const MAX_CLOCK_SKEW_SECONDS = 300;
const MAX_BODY_BYTES = 16_384;
const API_BASE_URL = "https://v3.football.api-sports.io";

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/health") {
      return json({ ok: true, service: "bet-ai-odds-collector", version: "1.0.0" });
    }

    const rawBody = request.method === "GET" ? "" : await readBody(request);
    const authorized = await verifyRequest(request, rawBody, env);
    if (!authorized) return json({ error: "unauthorized" }, 401);

    try {
      if (request.method === "POST" && url.pathname === "/v1/tracked-fixtures") {
        const payload = parseTrackFixture(rawBody);
        const result = await registerFixture(env, payload);
        return json(result, 201);
      }

      if (request.method === "GET" && url.pathname === "/v1/snapshots") {
        const cursor = parseNonNegativeInt(url.searchParams.get("after") ?? "0", "after");
        const limit = Math.min(
          parsePositiveInt(url.searchParams.get("limit") ?? "100", "limit"),
          250,
        );
        const snapshots = await env.DB.prepare(
          `SELECT id, tracked_fixture_id, api_fixture_id, snapshot_kind,
                  home_odd, draw_odd, away_odd, bookmaker, source,
                  captured_at, provider_updated_at, payload_sha256, details_json
             FROM odds_snapshots
            WHERE id > ?1
            ORDER BY id ASC
            LIMIT ?2`,
        ).bind(cursor, limit).all();
        return json({ snapshots: snapshots.results, next_cursor: lastId(snapshots.results, cursor) });
      }

      if (request.method === "POST" && url.pathname === "/v1/run") {
        return json(await runClosingCollection(env, "manual"));
      }

      return json({ error: "not_found" }, 404);
    } catch (error) {
      const message = error instanceof Error ? error.message : "unexpected_error";
      const status = error instanceof InputError ? 422 : 500;
      return json({ error: status === 422 ? message : "internal_error" }, status);
    }
  },

  async scheduled(_controller: ScheduledController, env: Env, ctx: ExecutionContext): Promise<void> {
    ctx.waitUntil(runClosingCollection(env, "scheduled"));
  },
} satisfies ExportedHandler<Env>;

class InputError extends Error {}

async function registerFixture(env: Env, payload: TrackFixtureRequest): Promise<Record<string, unknown>> {
  const now = new Date();
  const kickoff = new Date(payload.kickoff);
  if (!Number.isFinite(kickoff.getTime()) || kickoff <= now) {
    throw new InputError("kickoff must be a future ISO-8601 timestamp");
  }

  const timestamp = now.toISOString();
  await env.DB.prepare(
    `INSERT INTO tracked_fixtures (
       id, api_fixture_id, league_id, home_team, away_team, kickoff,
       preferred_bookmaker, status, created_at, updated_at
     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'active', ?8, ?8)
     ON CONFLICT(id) DO UPDATE SET
       api_fixture_id = COALESCE(excluded.api_fixture_id, tracked_fixtures.api_fixture_id),
       league_id = excluded.league_id,
       home_team = excluded.home_team,
       away_team = excluded.away_team,
       kickoff = excluded.kickoff,
       preferred_bookmaker = COALESCE(excluded.preferred_bookmaker, tracked_fixtures.preferred_bookmaker),
       updated_at = excluded.updated_at`,
  ).bind(
    payload.id,
    payload.api_fixture_id ?? null,
    payload.league_id,
    payload.home_team,
    payload.away_team,
    kickoff.toISOString(),
    payload.preferred_bookmaker ?? null,
    timestamp,
  ).run();

  let entry: Record<string, unknown> | null = null;
  if (payload.capture_entry !== false) {
    const fixture = await getTrackedFixture(env, payload.id);
    if (fixture) entry = await captureSnapshot(env, fixture, "entry");
  }
  return { ok: true, id: payload.id, entry };
}

async function runClosingCollection(
  env: Env,
  triggerType: "scheduled" | "manual",
): Promise<Record<string, unknown>> {
  const startedAt = new Date();
  const run = await env.DB.prepare(
    `INSERT INTO collection_runs (trigger_type, started_at, status)
     VALUES (?1, ?2, 'running') RETURNING id`,
  ).bind(triggerType, startedAt.toISOString()).first<{ id: number }>();
  if (!run) throw new Error("collection_run_not_created");

  let considered = 0;
  let recorded = 0;
  let errors = 0;
  try {
    const firstWindow = boundedInt(env.CLOSING_FIRST_WINDOW_MINUTES, 45, 5, 180);
    const retryWindow = boundedInt(env.CLOSING_RETRY_WINDOW_MINUTES, 15, 1, firstWindow);
    const deadline = new Date(startedAt.getTime() + firstWindow * 60_000).toISOString();
    const rows = await env.DB.prepare(
      `SELECT tf.id, tf.api_fixture_id, tf.league_id, tf.home_team, tf.away_team,
              tf.kickoff, tf.preferred_bookmaker, tf.closing_attempts
         FROM tracked_fixtures tf
        WHERE tf.status = 'active'
          AND tf.kickoff > ?1
          AND tf.kickoff <= ?2
          AND tf.closing_attempts < 2
          AND NOT EXISTS (
              SELECT 1 FROM odds_snapshots os
               WHERE os.tracked_fixture_id = tf.id AND os.snapshot_kind = 'closing'
          )
        ORDER BY tf.kickoff ASC
        LIMIT 20`,
    ).bind(startedAt.toISOString(), deadline).all<TrackedFixture>();

    for (const fixture of rows.results) {
      const minutesToKickoff = (new Date(fixture.kickoff).getTime() - startedAt.getTime()) / 60_000;
      if (fixture.closing_attempts === 1 && minutesToKickoff > retryWindow) continue;
      considered += 1;
      try {
        const result = await captureSnapshot(env, fixture, "closing");
        if (result?.recorded === true) recorded += 1;
      } catch {
        errors += 1;
      }
    }

    await expirePastFixtures(env, startedAt);
    await finishRun(env, run.id, "succeeded", considered, recorded, errors, null);
    return { ok: true, considered, recorded, errors };
  } catch (error) {
    const message = error instanceof Error ? error.message.slice(0, 500) : "unexpected_error";
    await finishRun(env, run.id, "failed", considered, recorded, errors + 1, message);
    throw error;
  }
}

async function captureSnapshot(
  env: Env,
  fixture: TrackedFixture,
  kind: "entry" | "closing",
): Promise<Record<string, unknown> | null> {
  const now = new Date();
  if (now >= new Date(fixture.kickoff)) return null;

  let apiFixtureId = fixture.api_fixture_id;
  if (!apiFixtureId) {
    apiFixtureId = await resolveApiFixtureId(env, fixture);
    if (!apiFixtureId) return { recorded: false, reason: "fixture_unresolved" };
    await env.DB.prepare(
      `UPDATE tracked_fixtures SET api_fixture_id = ?1, updated_at = ?2 WHERE id = ?3`,
    ).bind(apiFixtureId, now.toISOString(), fixture.id).run();
  }

  if (!(await consumeBudget(env, kind))) return { recorded: false, reason: "daily_budget_reserved" };
  const response = await apiFootballRequest<unknown>(env, "/odds", { fixture: String(apiFixtureId) });
  if (kind === "closing") await markClosingAttempt(env, fixture.id, now);
  const selection = selectMarket(response.body, fixture.preferred_bookmaker);
  if (!selection) return { recorded: false, reason: "market_unavailable" };

  const capturedAt = new Date().toISOString();
  const payloadSha256 = await sha256Hex(response.raw);
  const result = await env.DB.prepare(
    `INSERT OR IGNORE INTO odds_snapshots (
       tracked_fixture_id, api_fixture_id, snapshot_kind,
       home_odd, draw_odd, away_odd, bookmaker, source,
       captured_at, provider_updated_at, payload_sha256, details_json, created_at
     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'api_football_odds', ?8, ?9, ?10, ?11, ?8)`,
  ).bind(
    fixture.id,
    apiFixtureId,
    kind,
    selection.home,
    selection.draw,
    selection.away,
    selection.bookmaker,
    capturedAt,
    selection.providerUpdatedAt,
    payloadSha256,
    JSON.stringify(selection.details),
  ).run();

  if (kind === "closing" && result.meta.changes > 0) {
    await env.DB.prepare(
      `UPDATE tracked_fixtures SET status = 'completed', updated_at = ?1 WHERE id = ?2`,
    ).bind(capturedAt, fixture.id).run();
  }
  if (kind === "entry" && !fixture.preferred_bookmaker && result.meta.changes > 0) {
    // Lock closing collection to the same bookmaker used for the entry price.
    await env.DB.prepare(
      `UPDATE tracked_fixtures
          SET preferred_bookmaker = ?1, updated_at = ?2
        WHERE id = ?3 AND preferred_bookmaker IS NULL`,
    ).bind(selection.bookmaker, capturedAt, fixture.id).run();
  }
  return {
    recorded: result.meta.changes > 0,
    bookmaker: selection.bookmaker,
    captured_at: capturedAt,
    api_fixture_id: apiFixtureId,
    raw_odds: {
      HOME_WIN: selection.home,
      DRAW: selection.draw,
      AWAY_WIN: selection.away,
    },
    payload_sha256: payloadSha256,
  };
}

async function resolveApiFixtureId(env: Env, fixture: TrackedFixture): Promise<number | null> {
  const fixtureDate = fixture.kickoff.slice(0, 10);
  const homeKey = normalizeTeam(fixture.home_team);
  const awayKey = normalizeTeam(fixture.away_team);
  let match = await findCachedFixture(env, fixtureDate, fixture.league_id, homeKey, awayKey, fixture.kickoff);
  if (match) return match;

  const fetched = await env.DB.prepare(
    `SELECT fixture_date FROM provider_day_fetches WHERE fixture_date = ?1`,
  ).bind(fixtureDate).first();
  if (!fetched) {
    if (!(await consumeBudget(env, "entry"))) return null;
    const response = await apiFootballRequest<Record<string, unknown>>(env, "/fixtures", {
      date: fixtureDate,
      timezone: "UTC",
    });
    await cacheProviderFixtures(env, fixtureDate, response.body.response ?? []);
  }
  match = await findCachedFixture(env, fixtureDate, fixture.league_id, homeKey, awayKey, fixture.kickoff);
  return match;
}

async function findCachedFixture(
  env: Env,
  fixtureDate: string,
  leagueId: number,
  homeKey: string,
  awayKey: string,
  kickoff: string,
): Promise<number | null> {
  const expected = new Date(kickoff).getTime();
  const candidates = await env.DB.prepare(
    `SELECT api_fixture_id, kickoff FROM provider_fixtures
      WHERE fixture_date = ?1 AND league_id = ?2
        AND home_team_key = ?3 AND away_team_key = ?4`,
  ).bind(fixtureDate, leagueId, homeKey, awayKey).all<{ api_fixture_id: number; kickoff: string }>();
  const match = candidates.results.find(
    (candidate) => Math.abs(new Date(candidate.kickoff).getTime() - expected) <= 30 * 60_000,
  );
  return match?.api_fixture_id ?? null;
}

async function cacheProviderFixtures(
  env: Env,
  fixtureDate: string,
  rows: Record<string, unknown>[],
): Promise<void> {
  const now = new Date().toISOString();
  const statements: D1PreparedStatement[] = [];
  for (const row of rows.slice(0, 500)) {
    const fixture = asRecord(row.fixture);
    const league = asRecord(row.league);
    const teams = asRecord(row.teams);
    const home = asRecord(teams.home);
    const away = asRecord(teams.away);
    const id = positiveInt(fixture.id);
    const kickoff = typeof fixture.date === "string" ? fixture.date : null;
    const homeName = typeof home.name === "string" ? home.name : null;
    const awayName = typeof away.name === "string" ? away.name : null;
    if (!id || !kickoff || !homeName || !awayName) continue;
    statements.push(
      env.DB.prepare(
        `INSERT OR REPLACE INTO provider_fixtures (
           api_fixture_id, fixture_date, league_id, home_team, away_team,
           home_team_key, away_team_key, kickoff, cached_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)`,
      ).bind(
        id,
        fixtureDate,
        positiveInt(league.id),
        homeName,
        awayName,
        normalizeTeam(homeName),
        normalizeTeam(awayName),
        new Date(kickoff).toISOString(),
        now,
      ),
    );
  }
  for (let index = 0; index < statements.length; index += 75) {
    await env.DB.batch(statements.slice(index, index + 75));
  }
  await env.DB.prepare(
    `INSERT OR REPLACE INTO provider_day_fetches (fixture_date, fetched_at, fixture_count)
     VALUES (?1, ?2, ?3)`,
  ).bind(fixtureDate, now, statements.length).run();
}

async function apiFootballRequest<T>(
  env: Env,
  path: string,
  params: Record<string, string>,
): Promise<{ body: ApiResponse<T>; raw: string }> {
  if (!env.API_FOOTBALL_KEY) throw new Error("api_key_missing");
  const url = new URL(path, API_BASE_URL);
  for (const [key, value] of Object.entries(params)) url.searchParams.set(key, value);
  const response = await fetch(url, {
    headers: { "x-apisports-key": env.API_FOOTBALL_KEY },
  });
  const raw = await response.text();
  await updateProviderRemaining(env, response.headers.get("x-ratelimit-requests-remaining"));
  if (!response.ok) throw new Error(`api_football_http_${response.status}`);
  let body: ApiResponse<T>;
  try {
    body = JSON.parse(raw) as ApiResponse<T>;
  } catch {
    throw new Error("api_football_invalid_json");
  }
  if (body.errors && Object.keys(body.errors).length > 0) throw new Error("api_football_error_response");
  return { body, raw };
}

function selectMarket(body: ApiResponse<unknown>, preferredBookmaker?: string | null): OddsSelection | null {
  const first = asRecord(body.response?.[0]);
  const bookmakers = Array.isArray(first.bookmakers) ? first.bookmakers : [];
  const preferred = preferredBookmaker?.trim().toLocaleLowerCase("en-US") ?? null;
  const candidates: OddsSelection[] = [];
  for (const rawBookmaker of bookmakers) {
    const bookmaker = asRecord(rawBookmaker);
    const name = typeof bookmaker.name === "string" ? bookmaker.name.trim() : "";
    if (!name || (preferred && name.toLocaleLowerCase("en-US") !== preferred)) continue;
    const bets = Array.isArray(bookmaker.bets) ? bookmaker.bets : [];
    for (const rawBet of bets) {
      const bet = asRecord(rawBet);
      if (bet.name !== "Match Winner" || !Array.isArray(bet.values)) continue;
      const values = new Map<string, number>();
      for (const rawValue of bet.values) {
        const value = asRecord(rawValue);
        const odd = validOdd(value.odd);
        if (typeof value.value === "string" && odd) values.set(value.value, odd);
      }
      const home = values.get("Home");
      const draw = values.get("Draw");
      const away = values.get("Away");
      if (!home || !draw || !away) continue;
      candidates.push({
        home,
        draw,
        away,
        bookmaker: name,
        providerUpdatedAt: typeof first.update === "string" ? first.update : null,
        details: {
          provider_bookmaker_id: positiveInt(bookmaker.id),
          provider_bet_id: positiveInt(bet.id),
          overround_pct: round4((1 / home + 1 / draw + 1 / away - 1) * 100),
        },
      });
    }
  }
  return candidates.sort((a, b) => Number(a.details.overround_pct) - Number(b.details.overround_pct))[0] ?? null;
}

async function consumeBudget(env: Env, kind: "entry" | "closing"): Promise<boolean> {
  const day = new Date().toISOString().slice(0, 10);
  const dailyBudget = boundedInt(env.DAILY_API_BUDGET, 60, 1, 1000);
  const reserve = boundedInt(env.CLOSING_RESERVE, 30, 0, dailyBudget);
  const row = await env.DB.prepare(
    `SELECT requests_used FROM api_usage WHERE usage_date = ?1`,
  ).bind(day).first<{ requests_used: number }>();
  const used = row?.requests_used ?? 0;
  const ceiling = kind === "closing" ? dailyBudget : dailyBudget - reserve;
  if (used >= ceiling) return false;
  const now = new Date().toISOString();
  await env.DB.prepare(
    `INSERT INTO api_usage (usage_date, requests_used, updated_at)
     VALUES (?1, 1, ?2)
     ON CONFLICT(usage_date) DO UPDATE SET
       requests_used = requests_used + 1,
       updated_at = excluded.updated_at`,
  ).bind(day, now).run();
  return true;
}

async function updateProviderRemaining(env: Env, raw: string | null): Promise<void> {
  const remaining = raw === null ? null : Number.parseInt(raw, 10);
  if (remaining === null || !Number.isFinite(remaining)) return;
  const day = new Date().toISOString().slice(0, 10);
  await env.DB.prepare(
    `UPDATE api_usage SET provider_remaining = ?1, updated_at = ?2 WHERE usage_date = ?3`,
  ).bind(remaining, new Date().toISOString(), day).run();
}

async function markClosingAttempt(env: Env, id: string, when: Date): Promise<void> {
  await env.DB.prepare(
    `UPDATE tracked_fixtures
        SET closing_attempts = closing_attempts + 1,
            last_attempt_at = ?1,
            updated_at = ?1
      WHERE id = ?2`,
  ).bind(when.toISOString(), id).run();
}

async function expirePastFixtures(env: Env, now: Date): Promise<void> {
  await env.DB.prepare(
    `UPDATE tracked_fixtures
        SET status = CASE WHEN closing_attempts >= 2 THEN 'unresolved' ELSE 'expired' END,
            updated_at = ?1
      WHERE status = 'active' AND kickoff <= ?1`,
  ).bind(now.toISOString()).run();
}

async function finishRun(
  env: Env,
  id: number,
  status: "succeeded" | "failed",
  considered: number,
  recorded: number,
  errors: number,
  errorMessage: string | null,
): Promise<void> {
  await env.DB.prepare(
    `UPDATE collection_runs
        SET finished_at = ?1, status = ?2, fixtures_considered = ?3,
            snapshots_recorded = ?4, errors = ?5, error_message = ?6
      WHERE id = ?7`,
  ).bind(new Date().toISOString(), status, considered, recorded, errors, errorMessage, id).run();
}

async function getTrackedFixture(env: Env, id: string): Promise<TrackedFixture | null> {
  return env.DB.prepare(
    `SELECT id, api_fixture_id, league_id, home_team, away_team, kickoff,
            preferred_bookmaker, closing_attempts
       FROM tracked_fixtures WHERE id = ?1`,
  ).bind(id).first<TrackedFixture>();
}

function parseTrackFixture(raw: string): TrackFixtureRequest {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    throw new InputError("invalid JSON body");
  }
  const row = asRecord(value);
  const id = requiredText(row.id, "id", 120);
  const homeTeam = requiredText(row.home_team, "home_team", 120);
  const awayTeam = requiredText(row.away_team, "away_team", 120);
  const leagueId = positiveInt(row.league_id);
  if (!leagueId) throw new InputError("league_id must be a positive integer");
  const kickoff = requiredText(row.kickoff, "kickoff", 40);
  const apiFixtureId = row.api_fixture_id === undefined ? undefined : positiveInt(row.api_fixture_id);
  if (row.api_fixture_id !== undefined && !apiFixtureId) {
    throw new InputError("api_fixture_id must be a positive integer");
  }
  const preferred = optionalText(row.preferred_bookmaker, "preferred_bookmaker", 100);
  return {
    id,
    league_id: leagueId,
    home_team: homeTeam,
    away_team: awayTeam,
    kickoff,
    ...(apiFixtureId ? { api_fixture_id: apiFixtureId } : {}),
    ...(preferred ? { preferred_bookmaker: preferred } : {}),
    ...(typeof row.capture_entry === "boolean" ? { capture_entry: row.capture_entry } : {}),
  };
}

async function verifyRequest(request: Request, rawBody: string, env: Env): Promise<boolean> {
  const secret = env.CLIENT_HMAC_SECRET;
  if (!secret) return false;
  const timestamp = request.headers.get("x-betai-timestamp");
  const nonce = request.headers.get("x-betai-nonce");
  const signature = request.headers.get("x-betai-signature")?.toLowerCase();
  if (
    !timestamp ||
    !nonce ||
    !/^[a-zA-Z0-9_-]{24,128}$/.test(nonce) ||
    !signature ||
    !/^[a-f0-9]{64}$/.test(signature)
  ) return false;
  const unix = Number.parseInt(timestamp, 10);
  if (!Number.isFinite(unix) || Math.abs(Date.now() / 1000 - unix) > MAX_CLOCK_SKEW_SECONDS) return false;
  const url = new URL(request.url);
  const bodyHash = await sha256Hex(rawBody);
  const message = `${timestamp}.${nonce}.${request.method.toUpperCase()}.${url.pathname}${url.search}.${bodyHash}`;
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const expected = bytesToHex(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(message)));
  if (!timingSafeEqual(expected, signature)) return false;
  return reserveNonce(env.DB, nonce, unix);
}

async function reserveNonce(db: D1Database, nonce: string, issuedAt: number): Promise<boolean> {
  const now = Math.floor(Date.now() / 1000);
  await db.prepare("DELETE FROM request_nonces WHERE expires_at < ?1").bind(now).run();
  const result = await db.prepare(
    `INSERT OR IGNORE INTO request_nonces (nonce, issued_at, expires_at)
     VALUES (?1, ?2, ?3)`,
  ).bind(nonce, issuedAt, issuedAt + MAX_CLOCK_SKEW_SECONDS).run();
  return result.meta.changes === 1;
}

async function readBody(request: Request): Promise<string> {
  const declared = Number.parseInt(request.headers.get("content-length") ?? "0", 10);
  if (declared > MAX_BODY_BYTES) throw new InputError("request body too large");
  const body = await request.text();
  if (new TextEncoder().encode(body).byteLength > MAX_BODY_BYTES) {
    throw new InputError("request body too large");
  }
  return body;
}

function timingSafeEqual(left: string, right: string): boolean {
  if (left.length !== right.length) return false;
  let mismatch = 0;
  for (let index = 0; index < left.length; index += 1) {
    mismatch |= left.charCodeAt(index) ^ right.charCodeAt(index);
  }
  return mismatch === 0;
}

async function sha256Hex(value: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return bytesToHex(digest);
}

function bytesToHex(value: ArrayBuffer): string {
  return [...new Uint8Array(value)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function normalizeTeam(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/\p{Diacritic}/gu, "")
    .toLocaleLowerCase("en-US")
    .replace(/\b(fc|cf|afc|fk|sk|sc|ac)\b/g, " ")
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function validOdd(value: unknown): number | null {
  const number = typeof value === "number" ? value : Number.parseFloat(String(value));
  return Number.isFinite(number) && number > 1 && number <= 1000 ? round4(number) : null;
}

function positiveInt(value: unknown): number | null {
  if (typeof value === "boolean") return null;
  const number = typeof value === "number" ? value : Number.parseInt(String(value), 10);
  return Number.isSafeInteger(number) && number > 0 ? number : null;
}

function parsePositiveInt(value: string, field: string): number {
  const parsed = positiveInt(value);
  if (!parsed) throw new InputError(`${field} must be a positive integer`);
  return parsed;
}

function parseNonNegativeInt(value: string, field: string): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isSafeInteger(parsed) || parsed < 0) throw new InputError(`${field} must be non-negative`);
  return parsed;
}

function boundedInt(raw: string, fallback: number, minimum: number, maximum: number): number {
  const parsed = Number.parseInt(raw, 10);
  return Number.isSafeInteger(parsed) ? Math.min(maximum, Math.max(minimum, parsed)) : fallback;
}

function requiredText(value: unknown, field: string, maximum: number): string {
  if (typeof value !== "string" || !value.trim() || value.trim().length > maximum) {
    throw new InputError(`${field} must be a non-empty string up to ${maximum} characters`);
  }
  return value.trim();
}

function optionalText(value: unknown, field: string, maximum: number): string | undefined {
  if (value === undefined || value === null || value === "") return undefined;
  return requiredText(value, field, maximum);
}

function round4(value: number): number {
  return Math.round(value * 10_000) / 10_000;
}

function lastId(rows: unknown[], fallback: number): number {
  const last = asRecord(rows.at(-1));
  return positiveInt(last.id) ?? fallback;
}

function json(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), { status, headers: JSON_HEADERS });
}
