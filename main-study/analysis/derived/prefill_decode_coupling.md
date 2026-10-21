Engine view: decode log lines by prefill in their span (median ms per token per request; lines; no-graph share)

| Config | c | none: ms | none: lines | other: ms | other: lines | other: no-graph | other, graph lines only: ms | own: ms | own: lines | ambiguous: lines |
|---|---|---|---|---|---|---|---|---|---|---|
| R16-FP8 | 1 | 10.6 | 354–363 | 10.6–10.7 | 15–16 | 0.00 | 10.6–10.7 | 1550.5–1562.5 | 124 | 100–110 |
| R16-FP8 | 4 | 11.0 | 139–155 | 46.5–50.3 | 213–225 | 0.20–0.23 | 43.7–50.3 | 730.0–763.4 | 124–126 | 97–120 |
| R16-FP8 | 8 | 11.0–11.2 | 124–136 | 70.5–72.0 | 458–472 | 0.29–0.33 | 63.2–65.4 | 493.8–505.1 | 258–261 | 135–151 |
| R16-FP8 | 16 | 11.2 | 79–109 | 91.3–99.5 | 976–1019 | 0.38–0.41 | 71.8–79.8 | 290.7–301.2 | 545–547 | 189–203 |
| N4-FP8 | 1 | 7.5 | 303–322 | — | 0 | — | — | 7.6–7.8 | 87–93 | 217–231 |
| N4-FP8 | 4 | 8.5–8.6 | 66–74 | — | 0 | — | — | 10.7–10.9 | 54–58 | 82–87 |
| N4-FP8 | 8 | 7.8–8.6 | 58–63 | — | 0 | — | — | 12.9–13.2 | 72–75 | 77–82 |
| N4-FP8 | 16 | 9.7–9.8 | 19–24 | — | 0 | — | — | 16.0–16.2 | 113–117 | 78–82 |

Equal-span stratum: R16 lines whose span is exactly 1 s (median ms per token per request; lines), and the span length of all lines per class (median, max s)

| Config | c | none: ms at 1 s | none: lines at 1 s | other: ms at 1 s | other: lines at 1 s | other ÷ none at 1 s | none: span median, max s | other: span median, max s |
|---|---|---|---|---|---|---|---|---|
| R16-FP8 | 1 | 10.6 | 132–134 | 10.6–10.7 | 15–16 | 1.0 | 0, 1 | 1, 1 |
| R16-FP8 | 4 | 11.0 | 49–55 | 22.0–22.4 | 78–94 | 2.0 | 0, 1 | 2, 7–10 |
| R16-FP8 | 8 | 10.9–11.1 | 35–43 | 21.5–23.1 | 104–106 | 2.0–2.1 | 0, 1 | 3, 9–10 |
| R16-FP8 | 16 | 11.1–11.2 | 11–26 | 20.7–30.0 | 141–165 | 1.8–2.7 | 0, 1 | 4, 10 |

Client view: median TPOT ms (requests) by the share of a request's decode time during which another request was waiting for its first token

| Config | c | overlap 0 | overlap (0.0, 0.1] | overlap (0.1, 0.3] | overlap (0.3, 1.0] |
|---|---|---|---|---|---|
| R16-FP8 | 4 | 11.0 (27–28) | 10.7–11.0 (2) | 14.0 (0–2) | 41.8–43.9 (111–112) |
| R16-FP8 | 8 | 11.4–11.5 (19–23) | 11.0–11.4 (2–3) | 12.3–15.8 (0–1) | 80.8–85.3 (247–250) |
| R16-FP8 | 16 | 11.2 (19–21) | 11.1–11.7 (0–4) | 13.9–14.7 (1–2) | 127.9–134.5 (518–522) |
| N4-FP8 | 4 | 9.5–9.6 (31) | 10.0–10.1 (31) | 10.5–10.6 (70–71) | 12.8 (9–10) |
| N4-FP8 | 8 | 10.1–10.7 (17–19) | 11.1–11.3 (28–29) | 12.5–12.7 (124–126) | 14.5–14.7 (100–102) |
| N4-FP8 | 16 | 9.8 (21) | 11.3–11.5 (29–30) | 15.0–15.5 (155–160) | 18.8–19.1 (333–338) |

Overlap-0 requests: where their decode midpoint falls in the point (share of the span from the first send to the last completion) and how many requests were in flight then, counting the request itself (min–max over requests and launches)

| Config | c | decode midpoint share | in flight at midpoint |
|---|---|---|---|
| R16-FP8 | 4 | 0.01–1.00 | 1–4 |
| R16-FP8 | 8 | 0.62–1.00 | 1–8 |
| R16-FP8 | 16 | 0.57–1.00 | 2–15 |
| N4-FP8 | 4 | 0.02–1.00 | 1–4 |
| N4-FP8 | 8 | 0.03–1.00 | 1–8 |
| N4-FP8 | 16 | 0.46–0.99 | 2–15 |
