"""
alpha_loop.py — autonomous generate -> test -> learn -> repeat loop for WorldQuant BRAIN,
now with optional autonomous submission of alphas that clear a quality bar.

Setup (once), same folder as brain_engine.py:
    pip install requests python-dotenv google-genai pydantic
    .env needs three lines:
        WQ_EMAIL=you@example.com
        WQ_PASSWORD=your-password
        GEMINI_API_KEY=...

Run:
    python alpha_loop.py                      # Gemini writes the ideas
    python alpha_loop.py --try "<expression>" "<another>"
                                              # test your own expressions through the same
                                              # pipeline: simulate, check, submit if they qualify
    python alpha_loop.py --log-dir logs       # also keep the log in logs/alpha_loop_YYYY-MM-DD.log
                                              # (what the daily scheduled run uses)
    python alpha_loop.py --catch-up --log-dir logs
                                              # for extra triggers (start-up, wifi back, every few
                                              # hours): only runs while nothing is in for today
    To start a run by hand, Start-ScheduledTask -TaskName "Alpha Miner" runs it in the background with
    its log, the same as the scheduled run - it keeps going if you close the window.

Only one run at a time: a second run started while one is going exits straight away (alpha_loop.lock),
so a scheduled run and a manual one can never double-submit.

NOTE on the LLM library: this uses google-genai (import as `from google import genai`),
NOT google-generativeai (`import google.generativeai as genai`) - that package was
deprecated in August 2025 and is now archived/legacy. If you see import errors or silent
generation failures with the old package, that's why. Model defaults to gemini-3.8-flash
(current as of Sept 2026) - Google ships new Flash versions every few weeks, so this is
overridable via GEMINI_MODEL without touching code.

When Gemini is busy (503) or rate-limited (429), idea generation retries with backoff
(10s, 20s, 40s, 80s), then tries GEMINI_FALLBACK_MODEL once (default gemini-3.7-flash,
which Google's own 3.8 Flash docs name as the fully supported alternative). A failed
generation does NOT use up one of the MAX_ROUNDS - rounds only count when ideas actually
get tested. After MAX_GENERATION_FAILURES failed attempts in a row, the run stops cleanly;
nothing was simulated in that case, so no BRAIN quota was spent.

READ BEFORE YOU RUN THIS:
- It spends real BRAIN simulation quota and real Gemini API cost, unattended, up to the
  caps below (defaults: 5 rounds x 8 ideas = 40 simulations per run, plus follow-ups - and on a day
  with nothing submitted yet, up to 10 extra rounds within the 4-hour cap). Raise
  IDEAS_PER_ROUND / MAX_ROUNDS via env vars once you trust it, not before.
- This version can AUTONOMOUSLY SUBMIT alphas to BRAIN, at most MAX_SUBMISSIONS_PER_DAY
  (default 1) per BRAIN day, which starts at 3 AM US Eastern, not local midnight. An alpha
  qualifies when Sharpe >= 1.25, fitness >= 1.0 and every check passes, except SELF_CORRELATION
  while it reads PENDING: BRAIN only works that one out during submission, so it shows PENDING
  on every unsubmitted alpha. Before submitting, the loop asks BRAIN's free pre-submit
  self-correlation check and skips anything at 0.7 or above, so the daily slot isn't wasted - and, except
  for the end-of-day pick, holds anything at DAYTIME_MAX_SELF_CORRELATION (0.5) or above (see Near-copies).
  After submitting, it waits for BRAIN's verdict and only logs "YES" once BRAIN lists the
  alpha as ACTIVE. Anything unclear is logged as VERDICT_PENDING - confirm those with the
  dashboard's Check with BRAIN button. The submit and correlation endpoints follow community
  clients that recorded BRAIN's live traffic; BRAIN publishes no official API docs.
- Held alphas: one that clears the bar after today's submission is used is first run through BRAIN's
  free self-correlation check - too similar to your submitted alphas, it's logged TOO_CORRELATED;
  otherwise it's held as QUALIFIED_NOT_SUBMITTED. Every run (--try included) starts by re-checking
  every held alpha, slot free or not, since similarity grows as you submit more: one BRAIN already
  lists as submitted is marked YES, one where a check now FAILs is marked NOT_SUBMITTED, one that's
  become too similar is marked TOO_CORRELATED. If today's slot is free, the best remaining one goes in,
  as the rules below allow. If BRAIN can't be reached or the submit itself fails, the alpha stays
  queued for the next run. Re-checking stops after HELD_CHECK_BUDGET_SECONDS (default 300) per run.
- Auto-flip: when an alpha scores Sharpe -1.0 or worse, the loop also tests reverse(<expression>)
  right after it (up to MAX_FLIPS_PER_ROUND per round, default 3), since a strongly negative signal
  usually turns positive when flipped. Flips use extra simulations.
- Auto-smooth: when an alpha has Sharpe >= 1.25 but fitness under 1.0 and turnover above
  SMOOTH_MIN_TURNOVER (default 0.2) - or turnover over BRAIN's 70% limit with Sharpe >= 1.0
  (SMOOTH_MIN_SHARPE_OVER_LIMIT) - the loop also tests
  ts_decay_linear(<expression>, 10) right after it (up to MAX_SMOOTHS_PER_ROUND per round, default 3).
  Fitness divides by turnover, so trading less can lift it over the bar. The same goes for a strong Sharpe
  whose fitness misses the strong bar: with no failed check, turnover over SMOOTH_TOWARD_STRONG_MIN_TURNOVER
  (0.15), and fitness that would reach STRONG_MIN_FITNESS at BRAIN's 12.5% turnover floor with the same returns.
- Auto-repair: an alpha with Sharpe and fitness over the bar that FAILs one of BRAIN's fixable checks
  gets up to MAX_REPAIRS_PER_ALPHA (default 2) repaired versions tested right after it, at most
  MAX_REPAIRS_PER_ROUND (default 4) per round: CONCENTRATED_WEIGHT -> rank(...) or ts_backfill on its data
  fields; LOW_SUB_UNIVERSE_SHARPE -> rank within subindustry, neutralize against market-cap buckets, or trade
  the most liquid stocks on a faster version of the signal and the rest on a slower one;
  LOW_TURNOVER -> drop the outer ts_decay_linear. Repairs follow community write-ups of BRAIN's checks,
  and BRAIN's own result decides whether each worked. Fixes already tried are skipped, so a near-miss gets
  the next ones on a later run. An alpha whose Sharpe among the biggest stocks is under SUB_UNIVERSE_HOPELESS_SHARE
  (a third) of what BRAIN needs there isn't repaired at all: the signal barely works there, and no fix closes a
  gap that size. If the original is already too similar to your
  submitted alphas, it isn't repaired (the repair would be too). A near-miss failing a check with no
  automatic fix gets a log line naming that check, with BRAIN's value and limit for it when BRAIN gives them.
- Auto-sharpen: an alpha with Sharpe >= 1.25 but fitness under 1.0 whose turnover is already under 12.5%
  - where BRAIN's fitness formula stops giving credit for trading less - is re-tested without its outer
  ts_decay_linear/hump (up to MAX_SHARPENS_PER_ROUND, default 3), after the same free similarity check.
- Each normal run starts by re-examining up to MAX_PAST_REPAIRS (default 8) earlier near-misses whose
  repairs or sharpening haven't been tried.
- Never miss a day (the Gold leaderboard counts the days you submitted on): a normal run that still has
  nothing submitted for today after MAX_ROUNDS keeps going - up to MAX_EXTRA_ROUNDS (default 10) more rounds,
  within the MAX_RUNTIME_SECONDS cap - and stops as soon as one lands. Before each extra round the held
  queue gets another try. A submission only counts toward the day once BRAIN takes it: if BRAIN rejects
  one, the run goes on to the next good alpha. If a submit doesn't go through at all (connection dropped,
  BRAIN throttling), the alpha is held and tried again rather than lost.
- Reserve: today's first submission always goes in. Any more, up to MAX_SUBMISSIONS_PER_DAY, only when at
  least RESERVE_TARGET (default 3) other kinds of good alpha would still be waiting for the coming days.
  An alpha's kind is the BRAIN dataset its data fields come from (data_fields_cache.json; without it, the
  field name minus its numbers): alphas on the same data usually come back too similar to submit both, so
  near-copies count once. Only alphas BRAIN confirmed during this run count. RESERVE_TARGET=0 turns it off.
- Strong alphas first: an alpha that clears BRAIN's bar counts as strong with Sharpe >= STRONG_MIN_SHARPE (1.6),
  fitness >= STRONG_MIN_FITNESS (1.3) and annual returns above its worst drawdown (STRONG_MIN_RETURNS_TO_DRAWDOWN,
  1.0; 0 turns that part off). A strong alpha goes in straight away; one that only clears BRAIN's bar is held
  ("not strong"). Held alphas are ranked strong first, then by fitness, on BRAIN's latest numbers. If nothing
  has gone in by the end of the usual MAX_ROUNDS rounds - or the run stops early, or a --try run ends, or BRAIN's
  day is about to end - the run settles: an end-of-day pick from the held alphas that clear BRAIN's bar (see
  Portfolio check for how it's chosen) goes in, so the day can still count. Submissions beyond the day's first
  must be strong. Returns and drawdown are saved in the results file (an older file gets the two columns added
  once) and shown to the generator, which is told the strong targets.
- Near-copies: BRAIN accepts self-correlation up to 0.7, but every submission except the end-of-day pick must be
  under DAYTIME_MAX_SELF_CORRELATION (default 0.5) - a strong alpha straight away, an extra one, a held one.
  Near-copies of your own alphas add little to a combined score and keep your submissions looking alike. One
  in between - or one whose similarity BRAIN didn't report in time - is held instead, labelled "somewhat
  similar", and can still be the end-of-day pick. DAYTIME_MAX_SELF_CORRELATION=0.7 turns it off.
- Portfolio check: the Country Challenge's IS Score is "Combined Portfolio Performance based on an equal weighting
  of all submitted alphas". So before any alpha goes in, the loop reads the daily PnL of your submitted alphas
  (BRAIN's list of your ACTIVE alphas - or, failing that, the results file - and the PnL behind each alpha's
  chart), adds the new alpha's with equal weights, and lets it in only if the combined Sharpe rises. One that would
  lower it is held, labelled "would lower your combined portfolio", left out of the reserve and checked again each
  run. The end-of-day pick takes the held alpha that leaves the combined Sharpe highest (least similar breaks ties)
  and must raise it too - unless END_OF_DAY_MAX_PORTFOLIO_DROP (default 0) allows a small drop to keep the day's
  streak: 0.02 = up to 2%. If the PnL can't be read, the usual rules decide alone. PORTFOLIO_CHECK=0 turns it off.
- --backup, for a second scheduled run later in the day: if today's submission is already in and the
  reserve is full, it stops after re-checking the held queue; otherwise it mines like a normal run.
- Before/after check: before each submission, and for each alpha it holds, the loop asks BRAIN's free
  "before and after" check - how your merged performance would change with that alpha - for the competition in
  BRAIN_COMPETITION, or the only one BRAIN lists for you (each run logs your competitions, with your standing when
  BRAIN gives it), else for your account. BRAIN publishes no format for the reply, so it's only recorded - a
  one-line summary in the log, the full reply in logs/before_after_checks.jsonl - and decides nothing.
- Learning what doesn't work: any operator or field BRAIN rejects ('unknown variable/operator') is
  read back from the results file, removed from what the generator sees, and ideas still using it are
  dropped before they cost a simulation. Fields tried DEAD_END_MIN_TRIES (default 8) times without
  fitness reaching DEAD_END_MAX_FITNESS (default 0.5) are left out the same way; basic price/volume
  fields (close, volume, returns...) are exempt. Ideas using a data field BRAIN offers you that your field list
  (operators.json) leaves out are skipped too - usually old fields the generator remembers from your best past
  results; it needs data_fields_cache.json (from fetch_fields.py) to tell fields from other words.
- Variety: alphas built on the same kind of data usually come back too correlated to submit. Each
  field is labelled with its data type - BRAIN's own category from data_fields_cache.json (written by
  fetch_fields.py), or the dataset name in operators.json without it. The generator is told which types
  your submitted or queued alphas already use, and asked for VARIETY_EXTRA_IDEAS (default 6) spare
  ideas; the loop then keeps at most MAX_IDEAS_PER_TYPE (default 2) per type per round, and at most
  MAX_IDEAS_PER_CROWDED_TYPE (default 1) for types you've already submitted.
- Formula shape: each expression is reduced to its skeleton (smoothing wrapped around the outside,
  fields, numbers and groups stripped, any ts_*/group_* operator treated alike). Shapes your submitted, queued or too-similar alphas use - or that
  make up 30%+ of your last 60 simulations - count as overused: the generator is shown them with a list of
  other structures from your operator list, and the loop keeps at most MAX_IDEAS_PER_CROWDED_SHAPE
  (default 2) ideas per overused shape per round.
- Parallel simulations: up to PARALLEL_SIMULATIONS (default 3) run at once. If BRAIN answers that too
  many are running, the loop carries on with however many it already has going. Results are still
  handled one at a time, so submissions never overlap. Set PARALLEL_SIMULATIONS=1 for the old pace.
- Network resilience: every request to BRAIN has a 30-second timeout and is retried after a dropped
  connection (waiting 5s, 15s, 45s), and each Gemini request has a GEMINI_TIMEOUT_SECONDS (300) limit - the
  library's own default is none, so a dropped connection could freeze a run for hours. When something still
  can't get through, the loop asks whether BRAIN answers at all. If it does, that one idea is logged as
  "Couldn't reach BRAIN" and the loop moves on (after MAX_NETWORK_FAILURES, default 3, in a row it stops
  cleanly). If BRAIN doesn't answer, the connection itself is gone - wifi dropped, a laptop just woke up - so
  the run waits: it checks every minute for up to WAIT_FOR_CONNECTION_MINUTES (120) and then carries on where
  it left off, simulations already running on BRAIN included. Logging in waits the same way, and a login
  that failed only because the connection was down is no longer reported as identity verification. Time
  spent waiting, or with the computer asleep, doesn't count against MAX_RUNTIME_SECONDS.
- Laptops: while a run is going and the computer is plugged in, it asks Windows not to sleep (the screen can
  still turn off). On battery it lets the computer sleep as usual and carries on when it wakes.
- --catch-up, for extra scheduled triggers (start-up, reconnecting to the internet, every few hours): exits
  straight away if today's submission is already in, or if another run is going; otherwise it runs like a
  normal run and stops as soon as today's submission lands.
- End of BRAIN's day: a run with nothing in yet SETTLE_BEFORE_DAY_END_MINUTES (60) before BRAIN's day ends
  settles straight away - the end-of-day pick goes in - instead of waiting for a strong one it may not have
  time to find.
- Big-stocks check: BRAIN's numbers for it (the alpha's Sharpe among the largest stocks, and the limit there) are
  saved in the results file (sub_universe_sharpe, sub_universe_limit; an earlier near-miss gets them filled in
  when it's next looked at). The generator is told which ideas failed it - one example per idea, with every
  version of it (other windows, maturities, groups, smoothing or fixes) counting as the same idea - and asked to
  stop writing new versions of them, and those results are marked in the "best results" list it learns from.
- Padding: ideas like divide(x, add(y, 0.0001)) - a tiny number added so a division can't hit zero - are
  tested without it. Every alpha with it came back with an extra check that didn't PASS (BRAIN's units check:
  unitHandling is VERIFY, and a plain number has no units), and the loop only submits alphas whose every check
  passes. Follow-ups and repairs are built on the expression without it, and two expressions that differ only
  by padding count as the same idea when deciding what was already tried.
- Memory across runs: every round, the generator sees your best MEMORY_TOP_N (default 10) results
  from all runs so far (read from alpha_loop_results.csv), not just the last round.
- operators.json holds the full base-tier operator list (66 operators) transcribed from
  your BRAIN account. Its "fields" list is still the small 8-field starter set - add the
  data fields your account can use to widen what the generator can propose.
- "Unattended" has a real ceiling: BRAIN sessions expire (roughly 4 hours) and can
  require a one-time identity check to renew. This script saves its session to disk
  so a re-run doesn't need to log in again unnecessarily, but if BRAIN asks for
  verification mid-run with nobody there to click the link, it STOPS CLEANLY and
  logs why. It does not hang waiting for input that will never come.
"""
import argparse
import csv
import json
import logging
import math
import os
import re
import statistics
import time
from collections import Counter, deque, namedtuple
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx  # the HTTP library google-genai uses; its network errors are worth retrying
import requests
from google import genai
from google.genai import types
from pydantic import BaseModel

import brain_engine as be  # reuses: get_credentials, authenticate, submit_simulation,
                            # poll_simulation, fetch_alpha_metrics, looks_like_expired_session

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Guardrails: real quota and real money per round. Keep these low until you trust it. ----
IDEAS_PER_ROUND = int(os.environ.get("IDEAS_PER_ROUND", 8))
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", 5))
MAX_EXTRA_ROUNDS = int(os.environ.get("MAX_EXTRA_ROUNDS", 10))  # past MAX_ROUNDS, only while nothing is in for today
MAX_RUNTIME_SECONDS = int(os.environ.get("MAX_RUNTIME_SECONDS", 4 * 60 * 60))  # ~ BRAIN's own session lifetime
MAX_SUBMISSIONS_PER_DAY = int(os.environ.get("MAX_SUBMISSIONS_PER_DAY", 1))  # BRAIN's own submitter tools cap at 1/day
RESERVE_TARGET = int(os.environ.get("RESERVE_TARGET", 3))  # kinds of good alpha kept back before a 2nd one a day
# Strong alphas go in first; one that only clears BRAIN's bar waits until the usual rounds are done.
STRONG_MIN_SHARPE = float(os.environ.get("STRONG_MIN_SHARPE", 1.6))
STRONG_MIN_FITNESS = float(os.environ.get("STRONG_MIN_FITNESS", 1.3))
STRONG_MIN_RETURNS_TO_DRAWDOWN = float(os.environ.get("STRONG_MIN_RETURNS_TO_DRAWDOWN", 1.0))  # 0 turns this part off

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
# Google's own Gemini 3.8 Flash docs name 3.7 Flash as the fully supported alternative.
# Set GEMINI_FALLBACK_MODEL to an empty value in .env to turn the fallback off.
GEMINI_FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.7-flash")
RETRY_DELAYS_SECONDS = (10, 20, 40, 80)  # backoff when the main model is busy or rate-limited
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}  # worth waiting out; anything else won't fix itself
MAX_GENERATION_FAILURES = int(os.environ.get("MAX_GENERATION_FAILURES", 3))  # in a row, then stop the run
GENERATION_COOLDOWN_SECONDS = int(os.environ.get("GENERATION_COOLDOWN_SECONDS", 60))  # pause between failed attempts
# A time limit on each Gemini request. The library's default is none at all, so a connection that drops
# mid-request could leave a run waiting for hours - and while it waits, Windows skips the next scheduled run.
GEMINI_TIMEOUT_SECONDS = int(os.environ.get("GEMINI_TIMEOUT_SECONDS", 300))

# Submission bar - unchanged values, just named once so dashboard.py reads the same numbers.
SUBMIT_MIN_SHARPE = 1.25
SUBMIT_MIN_FITNESS = 1.0
# BRAIN's accepted turnover range: 1% to 70% of the book traded per day (reported as 0.01 to 0.70).
TURNOVER_MIN, TURNOVER_MAX = 0.01, 0.70
MAX_SELF_CORRELATION = 0.7  # BRAIN rejects a submission this correlated with your already-submitted alphas
# Stricter than BRAIN: every submission except the end-of-day pick (the one that keeps the day's streak) must be
# under this. Near-copies of your own alphas add little to a combined score and keep your alphas looking alike.
# One between this and 0.7 is held - it can still be an end-of-day pick, least similar first. 0.7 turns it off.
DAYTIME_MAX_SELF_CORRELATION = float(os.environ.get("DAYTIME_MAX_SELF_CORRELATION", 0.5))
CORR_POLL_MAX_SECONDS = int(os.environ.get("CORR_POLL_MAX_SECONDS", 300))
SUBMIT_POLL_MAX_SECONDS = int(os.environ.get("SUBMIT_POLL_MAX_SECONDS", 600))
CORRELATION_URL = "https://api.worldquantbrain.com/alphas/{alpha_id}/correlations/self"
SUBMIT_URL = "https://api.worldquantbrain.com/alphas/{alpha_id}/submit"
# BRAIN's free 'before and after' check: how your merged performance would change with one more alpha.
# The address follows WorldQuant's own ACE library; the reply's layout isn't published, so it's only recorded.
BEFORE_AFTER_URL = "https://api.worldquantbrain.com/users/self/alphas/{alpha_id}/before-and-after-performance"
BEFORE_AFTER_FILE = os.path.join("logs", "before_after_checks.jsonl")
BEFORE_AFTER_MAX_SECONDS = int(os.environ.get("BEFORE_AFTER_MAX_SECONDS", 120))
# The same check for one competition (the Country Challenge is part of one): BRAIN lists yours at COMPETITIONS_URL.
# The run asks it for the competition in BRAIN_COMPETITION, or the only one you're in, and only records the answer.
COMPETITIONS_URL = "https://api.worldquantbrain.com/users/self/competitions"
COMPETITION_BEFORE_AFTER_URL = ("https://api.worldquantbrain.com/competitions/{competition}/alphas/{alpha_id}"
                                "/before-and-after-performance")
BRAIN_COMPETITION = os.environ.get("BRAIN_COMPETITION", "").strip()

# Portfolio check. The leaderboard's IS Score is "Combined Portfolio Performance based on an equal weighting of all
# submitted alphas", so before an alpha goes in, its daily PnL is added to your submitted alphas' (equal weights) and
# it goes in only if the combined Sharpe rises. PORTFOLIO_CHECK=0 turns this off.
PORTFOLIO_CHECK = os.environ.get("PORTFOLIO_CHECK", "1").strip().lower() not in ("0", "false", "no", "off")
# How far the end-of-day pick - the one that keeps the day's streak - may lower that combined Sharpe: 0.02 = by up to
# 2%. At 0 it must raise it like any other, or the day goes without a submission.
END_OF_DAY_MAX_PORTFOLIO_DROP = float(os.environ.get("END_OF_DAY_MAX_PORTFOLIO_DROP", 0.0))
USER_ALPHAS_URL = "https://api.worldquantbrain.com/users/self/alphas"
PNL_URL = "https://api.worldquantbrain.com/alphas/{alpha_id}/recordsets/pnl"
PNL_MAX_SECONDS = 120     # per alpha, while BRAIN says its PnL is still being prepared
PORTFOLIO_MIN_DAYS = 250  # trading days all the alphas must share for a fair comparison (about a year)
ALPHA_LIST_PAGE = 100
MAX_ALPHA_LIST_PAGES = 30

# Auto-flip: a strongly negative Sharpe is a real signal pointing the wrong way, so test its reverse.
FLIP_SHARPE_AT_OR_BELOW = float(os.environ.get("FLIP_SHARPE_AT_OR_BELOW", -1.0))
MAX_FLIPS_PER_ROUND = int(os.environ.get("MAX_FLIPS_PER_ROUND", 3))  # caps the extra simulations flips add

# Auto-smooth: a good Sharpe held back by trading too much gets re-tested with linear decay.
SMOOTH_DECAY_DAYS = int(os.environ.get("SMOOTH_DECAY_DAYS", 10))
SMOOTH_MIN_TURNOVER = float(os.environ.get("SMOOTH_MIN_TURNOVER", 0.2))  # below this, decay can barely help fitness
SMOOTH_MIN_SHARPE_OVER_LIMIT = float(os.environ.get("SMOOTH_MIN_SHARPE_OVER_LIMIT", 1.0))  # turnover over 70%
MAX_SMOOTHS_PER_ROUND = int(os.environ.get("MAX_SMOOTHS_PER_ROUND", 3))  # caps the extra simulations smoothing adds
# ...and a strong Sharpe whose fitness misses the strong bar is smoothed too, when it trades more than this and
# trading at BRAIN's 12.5% fitness floor (same returns) would put fitness over STRONG_MIN_FITNESS.
SMOOTH_TOWARD_STRONG_MIN_TURNOVER = float(os.environ.get("SMOOTH_TOWARD_STRONG_MIN_TURNOVER", 0.15))

# Padding like add(x, 0.0001) - a tiny number added so a division can't hit zero - changes nothing that matters,
# but every alpha that had it came back with an extra check that didn't PASS (most likely BRAIN's units check:
# unitHandling is VERIFY, and a plain number has no units). Gemini's ideas are stripped of numbers up to this size.
PADDING_MAX = 0.001

# Network resilience: one dropped connection skips that idea; this many in a row stops the run cleanly.
MAX_NETWORK_FAILURES = int(os.environ.get("MAX_NETWORK_FAILURES", 3))
# ...unless the connection itself is gone (wifi dropped, laptop just woke up): then the run checks every
# CONNECTION_CHECK_SECONDS and carries on where it left off once BRAIN answers again, for up to
# WAIT_FOR_CONNECTION_MINUTES, and only then stops. Time spent waiting doesn't count against MAX_RUNTIME_SECONDS.
WAIT_FOR_CONNECTION_MINUTES = int(os.environ.get("WAIT_FOR_CONNECTION_MINUTES", 120))
CONNECTION_CHECK_SECONDS = 60
CONNECTION_PROBE_TIMEOUT = 15
SLEEP_GAP_SECONDS = 120  # a pause that ran this much over was the computer asleep (lid closed, Windows sleep)

# A run with nothing in for today this close to the end of BRAIN's day settles straight away (the end-of-day pick
# goes in) instead of waiting for a strong one it may not have time to find.
SETTLE_BEFORE_DAY_END_MINUTES = int(os.environ.get("SETTLE_BEFORE_DAY_END_MINUTES", 60))

# Auto-repair: a strong alpha (Sharpe and fitness over the bar) failing a fixable check gets
# re-tested with a known fix for that check. Each repair is one extra simulation.
MAX_REPAIRS_PER_ALPHA = int(os.environ.get("MAX_REPAIRS_PER_ALPHA", 2))
MAX_REPAIRS_PER_ROUND = int(os.environ.get("MAX_REPAIRS_PER_ROUND", 4))
MAX_PAST_REPAIRS = int(os.environ.get("MAX_PAST_REPAIRS", 8))  # earlier near-misses re-examined per run
# BRAIN's big-stocks check (LOW_SUB_UNIVERSE_SHARPE) re-runs an alpha on the largest, most liquid stocks and needs a
# Sharpe there of about 0.43x its own. One that gets under this share of that limit barely works there at all, and
# no fix closes a gap that size - so it isn't repaired.
SUB_UNIVERSE_HOPELESS_SHARE = float(os.environ.get("SUB_UNIVERSE_HOPELESS_SHARE", 1 / 3))
SMALL_STOCKS_EXAMPLES = 8  # ideas that failed the big-stocks check, shown to the generator as ones not to rework
# Auto-sharpen: below this turnover BRAIN's fitness formula gets no extra credit, so smoothing an alpha
# that's already this slow only dampens its returns.
FITNESS_TURNOVER_FLOOR = 0.125
MAX_SHARPENS_PER_ROUND = int(os.environ.get("MAX_SHARPENS_PER_ROUND", 3))
BACKFILL_DAYS = 120

# Learning what doesn't work: names BRAIN rejected, and fields that never got close.
DEAD_END_MIN_TRIES = int(os.environ.get("DEAD_END_MIN_TRIES", 8))
DEAD_END_MAX_FITNESS = float(os.environ.get("DEAD_END_MAX_FITNESS", 0.5))
BASIC_FIELDS = ("open", "high", "low", "close", "vwap", "volume", "returns", "adv20", "cap")  # never dropped

# Variety: alphas built on the same kind of data tend to be too correlated to submit together.
FIELDS_CACHE_FILE = "data_fields_cache.json"  # written by fetch_fields.py - holds BRAIN's category per field
MAX_IDEAS_PER_TYPE = int(os.environ.get("MAX_IDEAS_PER_TYPE", 2))  # per data type, per round
MAX_IDEAS_PER_CROWDED_TYPE = int(os.environ.get("MAX_IDEAS_PER_CROWDED_TYPE", 1))  # types you've already submitted
VARIETY_EXTRA_IDEAS = int(os.environ.get("VARIETY_EXTRA_IDEAS", 6))  # ask for spares, keep the most varied
# Formula shape: the operators nested the same way, whatever the fields, windows or groups.
MAX_IDEAS_PER_CROWDED_SHAPE = int(os.environ.get("MAX_IDEAS_PER_CROWDED_SHAPE", 2))  # per round
SHAPE_LOOKBACK = 60          # recent simulations looked at to spot an overused shape
SHAPE_CROWDED_SHARE = 0.3    # a shape making up this share of them counts as overused

# Parallel simulations: BRAIN runs a few of yours at once. If it says it's busy (429), the loop
# drops to however many were already running and keeps going.
PARALLEL_SIMULATIONS = max(1, int(os.environ.get("PARALLEL_SIMULATIONS", 3)))
SIM_CHECK_SECONDS = 5            # how often running simulations are checked
BUSY_WAIT_SECONDS = 15           # BRAIN busy and nothing of ours running: wait this long, then retry
MAX_BUSY_WAIT_SECONDS = 10 * 60  # ...for at most this long before giving up on that idea

# Memory across runs: how many of your best past results the generator sees each round.
MEMORY_TOP_N = int(os.environ.get("MEMORY_TOP_N", 10))

SETTINGS = {
    "instrumentType": "EQUITY", "region": "USA", "universe": "TOP3000",
    "delay": 1, "decay": 0, "neutralization": "SUBINDUSTRY", "truncation": 0.08,
    "pasteurization": "ON", "unitHandling": "VERIFY", "nanHandling": "ON",
    "language": "FASTEXPR", "visualization": False,
}

OPERATORS_FILE = "operators.json"
RESULTS_FILE = "alpha_loop_results.csv"  # separate from brain_engine.py's own results.csv
SESSION_FILE = "session_cookies.json"
SESSION_MIN_SECONDS = 30 * 60   # reuse a saved BRAIN login only if it has at least this long left
SESSION_RENEW_SECONDS = 5 * 60  # during a run, log in again this long before the login runs out
SUBMISSION_LOG_FILE = "submissions_log.json"  # {"YYYY-MM-DD": submissions BRAIN took that day} - shared by every run
HELD_PREFIX = "QUALIFIED_NOT_SUBMITTED"
HELD_LABEL = f"{HELD_PREFIX} (daily cap reached)"  # cleared the bar, waiting for a free submission slot
HELD_RETRY_LABEL = f"{HELD_PREFIX} (submit didn't go through - tried again next run)"
HELD_FOR_STRONGER_LABEL = f"{HELD_PREFIX} (not strong - kept in case a stronger one turns up)"
HELD_SIMILAR_LABEL = f"{HELD_PREFIX} (somewhat similar to your submitted alphas - kept for an end-of-day pick)"
HELD_PORTFOLIO_LABEL = f"{HELD_PREFIX} (would lower your combined portfolio - kept in case that changes)"
REPLACE_ATTEMPTS = 5  # Windows blocks swapping in the updated CSV while another program is reading it
HELD_CHECK_BUDGET_SECONDS = int(os.environ.get("HELD_CHECK_BUDGET_SECONDS", 300))  # per run, for re-checking held alphas

RESULTS_HEADER = [
    "round", "timestamp", "expression", "rationale", "status", "sharpe", "fitness",
    "turnover", "checks_passed", "checks_total", "alpha_id", "alpha_link", "error", "submitted",
    "returns", "drawdown", "sub_universe_sharpe", "sub_universe_limit",
]


class StopLoop(Exception):
    """Raised to unwind out of a round immediately - e.g. session can't be renewed unattended."""


class LoginFailed(RuntimeError):
    """Couldn't log in to BRAIN unattended (wrong password, or BRAIN wants identity verification)."""


class GenerationFailed(Exception):
    """Raised when no ideas could be generated this attempt (main model and fallback both
    failed). The main loop catches it and does NOT count the attempt as a round."""


# ---- Gemini structured-output schema (Pydantic - this is what the current SDK's docs use;
# the old TypedDict-based approach was tied to the deprecated library) ----
class Idea(BaseModel):
    expression: str
    rationale: str


class AlphaIdeas(BaseModel):
    ideas: list[Idea]


def load_reference():
    with open(OPERATORS_FILE, encoding="utf-8") as f:
        ref = json.load(f)
    return ref["operators"], ref["fields"]


def save_session(session: requests.Session):
    Path(SESSION_FILE).write_text(
        json.dumps(requests.utils.dict_from_cookiejar(session.cookies)), encoding="utf-8"
    )


def load_session(session: requests.Session) -> bool:
    if not Path(SESSION_FILE).exists():
        return False
    try:
        cookies = json.loads(Path(SESSION_FILE).read_text(encoding="utf-8"))
        session.cookies.update(requests.utils.cookiejar_from_dict(cookies))
        return True
    except Exception:
        return False


def session_seconds_left(session):
    """How long BRAIN says this login has left: GET /authentication -> token.expiry, the same check
    WorldQuant's own ACE library uses. 0 if BRAIN says it isn't logged in, None if it can't tell.
    (Asking about an alpha can't tell: logged out, BRAIN may answer 'not found' instead.)"""
    try:
        resp = session.get(be.AUTH_URL, timeout=be.REQUEST_TIMEOUT)
    except requests.RequestException:
        return None
    if resp.status_code in (401, 403) or be.looks_like_expired_session(resp):
        return 0.0
    if resp.status_code != 200:
        return None
    try:
        return max(0.0, float(resp.json()["token"]["expiry"]))
    except (ValueError, KeyError, TypeError):
        return None


def note_expiry(session):
    """Remember when this login runs out, so a long run can log in again before it does."""
    left = session_seconds_left(session)
    session.brain_expires_at = (time.time() + left) if left else None


def log_in(session, email, password) -> bool:
    """be.authenticate, made safe for runs nobody is watching: if BRAIN asks for identity verification
    and there's no keyboard to answer it, this returns False with a clear message instead of crashing."""
    try:
        ok = be.authenticate(session, email, password)
    except (EOFError, OSError, RuntimeError):
        logger.error("BRAIN wants you to confirm your identity, and nobody is here to do it. Run "
                     "python alpha_loop.py by hand once and open the link it prints.")
        return False
    if ok:
        save_session(session)
        note_expiry(session)
    return ok


def get_authenticated_session(email: str, password: str) -> requests.Session:
    session = requests.Session()
    if load_session(session):
        left = session_seconds_left(session)
        if left is not None and left >= SESSION_MIN_SECONDS:
            session.brain_expires_at = time.time() + left
            logger.info("Reused saved session (%d minutes left on it) - skipped login.", left // 60)
            return session
        logger.info("Saved session %s - logging in fresh.",
                    "couldn't be confirmed" if left is None else "has expired" if not left
                    else f"has under {SESSION_MIN_SECONDS // 60} minutes left")
        session.cookies.clear()

    if not log_in(session, email, password):
        raise LoginFailed("Could not log in to BRAIN - stopping before using any quota.")
    return session


def keep_logged_in(session, email, password):
    """Log in again shortly before BRAIN's login runs out, instead of waiting for requests to fail."""
    expires_at = getattr(session, "brain_expires_at", None)
    if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
        return
    if time.time() < expires_at - SESSION_RENEW_SECONDS:
        return
    logger.info("BRAIN login runs out in under %d minutes - logging in again now.", SESSION_RENEW_SECONDS // 60)
    session.cookies.clear()
    if not log_in_again(session, email, password):
        raise StopLoop("BRAIN needs re-authentication (likely identity verification) and no one is here to complete it.")


# ---- Riding out a lost connection (wifi dropped, laptop asleep) ----

OUTAGE_SECONDS = 0.0  # this run's time waiting for the connection or asleep - not counted against MAX_RUNTIME_SECONDS


def no_connection() -> StopLoop:
    return StopLoop(f"No connection to BRAIN for {WAIT_FOR_CONNECTION_MINUTES} minutes - stopped cleanly; "
                    f"everything so far is saved. The next scheduled run carries on.")


def brain_reachable() -> bool:
    """Whether BRAIN answers at all. Any reply counts, even 'not logged in' - only a failed connection doesn't."""
    try:
        requests.get(be.AUTH_URL, timeout=CONNECTION_PROBE_TIMEOUT)
        return True
    except requests.RequestException:
        return False


def wait_for_connection(what: str) -> bool:
    """If BRAIN can't be reached, check every CONNECTION_CHECK_SECONDS until it can, for up to
    WAIT_FOR_CONNECTION_MINUTES. True as soon as BRAIN answers (straight away if it already does)."""
    global OUTAGE_SECONDS
    if brain_reachable():
        return True
    began = time.time()
    logger.warning("Lost the connection to BRAIN %s - waiting for it to come back (checking every %ds, for up to "
                   "%d minutes).", what, CONNECTION_CHECK_SECONDS, WAIT_FOR_CONNECTION_MINUTES)
    try:
        for _ in range(max(1, WAIT_FOR_CONNECTION_MINUTES * 60 // CONNECTION_CHECK_SECONDS)):
            time.sleep(CONNECTION_CHECK_SECONDS)
            if brain_reachable():
                logger.info("The connection is back after %d minute(s) - carrying on where the run left off.",
                            round((time.time() - began) / 60))
                return True
        logger.error("Still no connection to BRAIN after %d minutes.", WAIT_FOR_CONNECTION_MINUTES)
        return False
    finally:
        OUTAGE_SECONDS += time.time() - began


def nap(seconds) -> float:
    """time.sleep that notices the computer sleeping through it: a pause that ran SLEEP_GAP_SECONDS or more over
    was the computer asleep, and that time isn't counted against MAX_RUNTIME_SECONDS. Returns the seconds the
    computer was away (0 normally)."""
    global OUTAGE_SECONDS
    began = time.time()
    time.sleep(seconds)
    away = time.time() - began - seconds
    if away < SLEEP_GAP_SECONDS:
        return 0.0
    OUTAGE_SECONDS += away
    logger.info("The computer seems to have been asleep for about %d minute(s) - carrying on.", round(away / 60))
    return away


def log_in_again(session, email, password) -> bool:
    """log_in, but a login that failed only because the connection is gone waits for it to come back and tries
    once more (raising StopLoop if it never does). False means BRAIN itself said no - identity verification
    nobody is here to answer, or a wrong password."""
    if log_in(session, email, password):
        return True
    if brain_reachable():
        return False
    if not wait_for_connection("while logging in again"):
        raise no_connection()
    return log_in(session, email, password)


def connect(email: str, password: str) -> requests.Session:
    """Log in for a run - first waiting out a lost connection (a laptop that just woke up, wifi that dropped),
    and once more if it drops during the login itself."""
    if not wait_for_connection("before logging in"):
        raise LoginFailed(f"No connection to BRAIN for {WAIT_FOR_CONNECTION_MINUTES} minutes - stopping before "
                          f"using any quota. The next scheduled run tries again.")
    try:
        return get_authenticated_session(email, password)
    except LoginFailed:
        if brain_reachable() or not wait_for_connection("while logging in"):
            raise
        return get_authenticated_session(email, password)


# ---- Laptops: don't let Windows sleep in the middle of a run while it's plugged in ----

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
AWAKE = False  # whether this run has asked Windows to stay awake


def on_mains_power():
    """True when Windows says the computer is plugged in, False on battery, None when it can't tell
    (or this isn't Windows)."""
    try:
        import ctypes

        class PowerStatus(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

        status = PowerStatus()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return None
        return {0: False, 1: True}.get(status.ACLineStatus)
    except Exception:
        return None


def set_execution_state(flags: int) -> bool:
    """Windows' SetThreadExecutionState. False where there's no such thing (not Windows). Its reply is the previous
    state, which can be 0 the first time, so any call that went through counts as done."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))
        return True
    except Exception:
        return False


def keep_awake():
    """While a run is going and the computer is plugged in, ask Windows not to sleep (the screen can still turn
    off). On battery, let it sleep as usual - the run carries on when it wakes. Called every round and every
    simulation check, so plugging in or unplugging mid-run is noticed. Does nothing on other systems."""
    global AWAKE
    want = on_mains_power() is True
    if want == AWAKE:
        return
    if not set_execution_state(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if want else 0)):
        return
    AWAKE = want
    logger.info("Plugged in - keeping the computer awake until this run is done." if want else
                "On battery - letting the computer sleep as usual; the run carries on when it wakes.")


def let_sleep():
    """The run is over: Windows may sleep again whenever it likes."""
    global AWAKE
    if AWAKE and set_execution_state(ES_CONTINUOUS):
        AWAKE = False


def already_tried() -> set:
    """Every expression tried so far - each also without its add(x, 0.0001) padding, so an idea that differs from
    an earlier one only by the padding counts as already tried."""
    if not Path(RESULTS_FILE).exists():
        return set()
    with open(RESULTS_FILE, encoding="utf-8") as f:
        tried = {row["expression"] for row in csv.DictReader(f)}
    return tried | {strip_padding(e) for e in tried}


def note_tried(avoid: set, expression: str):
    avoid.add(expression)
    avoid.add(strip_padding(expression))


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def brain_day(now_utc=None) -> date:
    """BRAIN's day starts at 3 AM US Eastern, not at your local midnight. Worked out from the US
    daylight-saving rules directly, so it needs no time-zone database (Windows doesn't ship one)."""
    now_utc = now_utc or datetime.now(timezone.utc)

    def nth_sunday(month, n):
        first_weekday = date(now_utc.year, month, 1).weekday()  # Monday=0 ... Sunday=6
        return 1 + (6 - first_weekday) % 7 + 7 * (n - 1)

    dst_start = datetime(now_utc.year, 3, nth_sunday(3, 2), 7, tzinfo=timezone.utc)   # 2 AM EST
    dst_end = datetime(now_utc.year, 11, nth_sunday(11, 1), 6, tzinfo=timezone.utc)   # 2 AM EDT
    eastern = timedelta(hours=-4) if dst_start <= now_utc < dst_end else timedelta(hours=-5)
    return (now_utc + eastern - timedelta(hours=3)).date()


def brain_day_ends_in(now_utc=None) -> timedelta:
    """How long until BRAIN's day rolls over (3 AM US Eastern: 07:00 UTC in summer time, 08:00 UTC in winter)."""
    now_utc = now_utc or datetime.now(timezone.utc)
    following = brain_day(now_utc) + timedelta(days=1)
    for hour in (7, 8):
        rollover = datetime(following.year, following.month, following.day, hour, tzinfo=timezone.utc)
        if brain_day(rollover) == following:
            return rollover - now_utc
    return timedelta(days=1)  # unreachable: one of the two hours is always the rollover


def submissions_today() -> int:
    if not Path(SUBMISSION_LOG_FILE).exists():
        return 0
    try:
        data = json.loads(Path(SUBMISSION_LOG_FILE).read_text(encoding="utf-8"))
    except Exception:
        return 0
    return data.get(str(brain_day()), 0)


def record_submission():
    data = {}
    if Path(SUBMISSION_LOG_FILE).exists():
        try:
            data = json.loads(Path(SUBMISSION_LOG_FILE).read_text(encoding="utf-8"))
        except Exception:
            data = {}
    key = str(brain_day())
    data[key] = data.get(key, 0) + 1
    Path(SUBMISSION_LOG_FILE).write_text(json.dumps(data), encoding="utf-8")


def as_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def returns_note(row) -> str:
    """' returns=... drawdown=...' when BRAIN reported them, flagging an alpha that lost more in its worst
    stretch than it made over the year."""
    returns, drawdown = as_float(row.get("returns")), as_float(row.get("drawdown"))
    if returns is None or drawdown is None:
        return ""
    return f" returns={row['returns']} drawdown={row['drawdown']}" + (
        " RETURNS BELOW DRAWDOWN" if returns <= abs(drawdown) else "")


def turnover_note(row) -> str:
    turnover = as_float(row.get("turnover"))
    if turnover is None:
        return ""
    if turnover > TURNOVER_MAX:
        return f" TURNOVER TOO HIGH (max {TURNOVER_MAX})"
    if turnover < TURNOVER_MIN:
        return f" TURNOVER TOO LOW (min {TURNOVER_MIN})"
    return ""


def summarize_round(rows: list) -> str:
    """Condense one round's results into text the next generation prompt can use."""
    if not rows:
        return "This is round 1 - no prior results yet."
    completed = [r for r in rows if r["status"] == "COMPLETE"]
    if not completed:
        return f"Last round: {len(rows)} attempts, none completed - check the error column pattern before trying similar shapes again."
    ranked = sorted(completed, key=lambda r: safe_float(r["fitness"]), reverse=True)
    lines = ["Last round's results, best fitness first:"]
    for r in ranked:
        lines.append(
            f"- {r['expression']} -> sharpe={r['sharpe']} fitness={r['fitness']} "
            f"turnover={r['turnover']}{turnover_note(r)}{returns_note(r)} checks={r['checks_passed']}/{r['checks_total']} "
            f"submitted={r.get('submitted', 'NO')}"
            + (f" not_passing=[{r['failing_checks']}]" if r.get("failing_checks") else "")
        )
    failed = [r for r in rows if r["status"] != "COMPLETE"]
    if failed:
        lines.append(f"{len(failed)} other attempts failed outright (bad expression, timeout, etc) - avoid whatever pattern caused that.")
    return "\n".join(lines)


def completed_history() -> list:
    """Every completed simulation from every run so far, read from the results CSV."""
    if not Path(RESULTS_FILE).exists():
        return []
    with open(RESULTS_FILE, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)
                if r.get("status") == "COMPLETE" and as_float(r.get("fitness")) is not None]


def pool_tag(row) -> str:
    """Marks results that are already in (or headed for) your submitted pool."""
    submitted = (row.get("submitted") or "").strip()
    if submitted == "YES" or submitted.startswith("VERDICT_PENDING"):
        return " [submitted]"
    if submitted.startswith(HELD_PREFIX):
        return " [queued]"
    if submitted.startswith("TOO_CORRELATED"):
        return " [too similar to a submitted alpha]"
    return ""


# ---- Learning what doesn't work ----

_REJECTED = re.compile(r'unknown (?:variable|operator) "([^"]+)"', re.IGNORECASE)


def rejected_names() -> set:
    """Operators and fields BRAIN refused on your account ('unknown variable/operator "x"'), read
    from the error column of every run so far."""
    if not Path(RESULTS_FILE).exists():
        return set()
    with open(RESULTS_FILE, encoding="utf-8", newline="") as f:
        return {name for r in csv.DictReader(f) for name in _REJECTED.findall(r.get("error") or "")}


def dead_end_fields() -> dict:
    """{field: (simulations, best fitness)} for fields tried DEAD_END_MIN_TRIES+ times without fitness
    ever reaching DEAD_END_MAX_FITNESS. Basic price/volume fields are exempt - they're ingredients."""
    stats = {}
    for r in completed_history():
        fitness = as_float(r.get("fitness"))
        for name in {n for _, _, n in data_fields_in(r["expression"])}:
            tries, best = stats.get(name, (0, fitness))
            stats[name] = (tries + 1, max(best, fitness))
    return {name: (tries, best) for name, (tries, best) in stats.items()
            if name not in BASIC_FIELDS and tries >= DEAD_END_MIN_TRIES and best < DEAD_END_MAX_FITNESS}


def uses_any(expression: str, names) -> bool:
    return bool(set(_WORD.findall(_blank_quotes(expression))) & set(names))


def avoid_note(rejected, dead: dict) -> str:
    lines = []
    if rejected:
        lines.append("BRAIN rejected these names on your account - never use them: "
                     + ", ".join(sorted(rejected)) + ". (Write comparisons with >, <, == instead.)")
    if dead:
        lines.append(f"These fields were tried {DEAD_END_MIN_TRIES}+ times and never reached fitness "
                     f"{DEAD_END_MAX_FITNESS}, so they're left out: "
                     + ", ".join(f"{n} ({t} tries, best {b:.2f})" for n, (t, b) in sorted(dead.items())) + ".")
    return ("DON'T USE:\n" + "\n".join(f"- {line}" for line in lines)) if lines else ""


# ---- Variety: spread ideas across data types ----

def field_types(fields) -> dict:
    """{field name: data type} for the signal fields the generator can use. The data type is BRAIN's
    own category for the field (Option, Fundamental, Analyst...), read from data_fields_cache.json;
    without that file, the dataset name fetch_fields.py keeps in [brackets] in each description.
    Group fields (sector, industry...) are left out - every idea uses them for grouping."""
    categories = {}
    try:
        cache = json.loads(Path(FIELDS_CACHE_FILE).read_text(encoding="utf-8"))
        for f in cache.get("matrix_fields") or []:
            cat = f.get("category")
            name = (cat.get("name") or cat.get("id")) if isinstance(cat, dict) else cat
            if f.get("id") and name:
                categories[f["id"]] = str(name)
    except (OSError, ValueError, AttributeError):
        pass
    found = {}
    for f in fields:
        name, desc = f.get("name"), (f.get("description") or "").strip()
        if not name or desc.startswith("group field"):
            continue
        if name in categories:
            found[name] = categories[name]
        else:
            tag = re.search(r"\[([^\[\]]+)\]$", desc)
            if tag:
                found[name] = tag.group(1)
    return found


def expression_types(expression: str, type_map: dict) -> set:
    """The data types an expression uses. Basic price/volume data (close, returns, volume...) goes into most
    formulas as an ingredient, so it only counts when the formula uses nothing else."""
    words = {w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression) if w in type_map}
    basic = {type_map[b] for b in BASIC_FIELDS if b in type_map}
    main = {type_map[w] for w in words if w not in BASIC_FIELDS and type_map[w] not in basic}
    return main or {type_map[w] for w in words}


def crowded_types(type_map: dict) -> Counter:
    """Data types your submitted, queued or too-similar alphas are built on, with how many of each."""
    counts = Counter()
    if not type_map or not Path(RESULTS_FILE).exists():
        return counts
    with open(RESULTS_FILE, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if pool_tag(r):
                counts.update(expression_types(r.get("expression") or "", type_map))
    return counts


def per_type_cap(n: int, type_map: dict) -> int:
    """Ideas allowed per data type - loosened if there are too few types to fill a round otherwise."""
    return max(MAX_IDEAS_PER_TYPE, math.ceil(n / max(1, len(set(type_map.values())))))


def type_quota(type_map: dict, crowded: Counter, asked: int) -> dict:
    """How many of the `asked` ideas to request per data type. Types your submitted or queued alphas already
    use get MAX_IDEAS_PER_CROWDED_TYPE each; the rest is dealt out over the other types one at a time - types
    with more fields first, none more than twice its number of fields - so the ideas can fill a round
    within the per-type limits instead of piling onto one type."""
    fields_per_type = Counter(type_map.values())
    order = sorted(fields_per_type, key=lambda t: (t in crowded, -fields_per_type[t], t))
    room = {t: MAX_IDEAS_PER_CROWDED_TYPE if t in crowded else 2 * fields_per_type[t] for t in order}
    quota = Counter()
    while sum(quota.values()) < asked and any(quota[t] < room[t] for t in order):
        for t in order:
            if quota[t] < room[t] and sum(quota.values()) < asked:
                quota[t] += 1
    return {t: quota[t] for t in order if quota[t]}


def variety_rule(type_map: dict, crowded: Counter, n: int, asked: int = 0) -> str:
    all_types = sorted(set(type_map.values()))
    if len(all_types) < 2:
        return ""
    lines = ["VARIETY RULE - this is what gets new alphas accepted. BRAIN won't take an alpha whose "
             "self-correlation with your submitted alphas is 0.7 or more, and alphas built on the same "
             "kind of data usually are. Every field above is labelled with its data type."]
    if crowded:
        used = ", ".join(f"{t} ({c})" for t, c in crowded.most_common())
        lines.append(f"- Your submitted or queued alphas already use: {used}. At most "
                     f"{MAX_IDEAS_PER_CROWDED_TYPE} idea(s) this round may use each of these.")
    fresh = [t for t in all_types if t not in crowded]
    if fresh:
        lines.append(f"- Not used yet - favour these: {', '.join(fresh)}.")
    quota = type_quota(type_map, crowded, asked or n)
    rest = (asked or n) - sum(quota.values())
    spread = ", ".join(f"{t} {q}" for t, q in quota.items()) + (f", and {rest} on any type" if rest > 0 else "")
    lines.append(f"- Write exactly this many ideas per data type: {spread}. Put your best idea for each type first: "
                 f"only the first {per_type_cap(n, type_map)} of a type"
                 + (f" ({MAX_IDEAS_PER_CROWDED_TYPE} for the types your alphas already use)" if crowded else "")
                 + " are sure to be tested, the rest are spares.")
    lines.append("- Carry the shapes that scored well (operators, windows, grouping, decay) over to other data types "
                 "instead of repeating the same data.")
    return "\n".join(lines)


# ---- Variety of formula shape ----

SMOOTHERS = ("ts_decay_linear", "hump")


def strip_smoothing(expression: str) -> str:
    """The expression without any ts_decay_linear / hump wrapped around the outside."""
    expr = expression.strip()
    while True:
        call = outer_call(expr)
        if not call or call[0] not in SMOOTHERS or not call[1][0].strip():
            return expr
        expr = call[1][0].strip()


def skeleton(expression: str) -> str:
    """The formula's shape with the specifics stripped out: smoothing wrapped around the outside
    (ts_decay_linear, hump) is ignored - it controls turnover, it isn't the idea - then data fields -> F,
    groups -> G, numbers -> N, any ts_* operator -> ts_*, any group_* operator -> group_*. Two alphas with
    the same skeleton are the same idea with different data, windows or smoothing."""
    expression = strip_smoothing(expression)
    group_spans = list(_group_calls(expression))
    cuts = [(s, e, "G") for s, e in group_spans]
    cuts += [(s, e, "F") for s, e, _ in data_fields_in(expression)
             if not any(gs <= s and e <= ge for gs, ge in group_spans)]
    out = expression
    for s, e, token in sorted(cuts, reverse=True):
        out = out[:s] + token + out[e:]
    out = _QUOTED.sub("Q", out)
    out = re.sub(r"(?<![A-Za-z_])\d+(\.\d+)?([eE]-?\d+)?", "N", out)
    out = re.sub(r"\bts_\w+", "ts_*", out)
    out = re.sub(r"\bgroup_\w+", "group_*", out)
    return re.sub(r"\s+", "", out)


def crowded_shapes() -> dict:
    """{skeleton: example expression} for shapes your submitted, queued or too-similar alphas use, plus
    any shape that makes up SHAPE_CROWDED_SHARE or more of your last SHAPE_LOOKBACK simulations."""
    if not Path(RESULTS_FILE).exists():
        return {}
    with open(RESULTS_FILE, encoding="utf-8", newline="") as f:
        done = [r for r in csv.DictReader(f) if r.get("status") == "COMPLETE" and r.get("expression")]
    shapes = {}
    for r in done:
        if pool_tag(r):
            shapes[skeleton(r["expression"])] = r["expression"]
    recent = done[-SHAPE_LOOKBACK:]
    if len(recent) >= 10:
        counts = Counter(skeleton(r["expression"]) for r in recent)
        for r in reversed(recent):  # newest example, but a submitted alpha's from above wins
            shape = skeleton(r["expression"])
            if counts[shape] / len(recent) >= SHAPE_CROWDED_SHARE:
                shapes.setdefault(shape, r["expression"])
    return shapes


SHAPE_IDEAS = (  # (operators needed, suggestion) - only suggested if your account has those operators
    (("rank", "divide"), "a ratio or spread of two fields, ranked: rank(divide(x, y)) or rank(x - y)"),
    (("ts_corr", "rank"), "how two fields move together: -ts_corr(rank(x), rank(y), d)"),
    (("ts_regression",), "what's left after regressing one field on another: ts_regression(y, x, d)"),
    (("trade_when",), "a position that only updates when a condition holds: trade_when(condition, signal, -1)"),
    (("ts_delta", "ts_std_dev"), "a change scaled by its own volatility: ts_delta(x, d) / ts_std_dev(x, d)"),
    (("rank",), "a blend of two different data types: rank(x) + rank(y)"),
    (("ts_rank",), "where today's value sits in its own history: ts_rank(x, d)"),
    (("if_else",), "switching between two signals: if_else(condition, signal_a, signal_b)"),
)


def shape_rule(shapes: dict, operators) -> str:
    if not shapes:
        return ""
    names = {o.get("name") for o in operators or []}
    examples = "; ".join(list(shapes.values())[:3])
    lines = ["SHAPE RULE - vary the formula's structure, not just its data. These shapes are already "
             f"overused (same operators nested the same way, whatever the fields, windows, groups or "
             f"smoothing) - e.g. {examples}. At most {MAX_IDEAS_PER_CROWDED_SHAPE} ideas this round may "
             f"use any one of them."]
    ideas = [text for needed, text in SHAPE_IDEAS if not names or all(op in names for op in needed)]
    if ideas:
        lines.append("Other structures to try:\n" + "\n".join(f"- {text}" for text in ideas))
    lines.append("Smoothing is NOT part of the shape - keep using it. Wrap new structures in "
                 "ts_decay_linear(x, 10) to ts_decay_linear(x, 20), or hump(x, hump=0.01): most raw "
                 "structures trade far more than BRAIN's 70% turnover limit and get rejected without it.")
    return "\n".join(lines)


def label_fields(fields, type_map: dict) -> list:
    return [dict(f, description=f"[type: {type_map[f['name']]}] {f.get('description', '')}")
            if f.get("name") in type_map else f for f in fields]


def pick_varied(ideas, type_map: dict, crowded: Counter, n: int, overused_shapes=()) -> list:
    """Keep up to n ideas, in Gemini's order, without going over the per-type limits or using an overused
    formula shape more than MAX_IDEAS_PER_CROWDED_SHAPE times. If the limits leave free slots, ideas they held
    back are let in after all - each data type up to twice its usual limit - rather than run a half-empty round."""
    cap = per_type_cap(n, type_map)

    def limit(t, stretch=1):
        return stretch * (MAX_IDEAS_PER_CROWDED_TYPE if t in crowded else cap)

    used, shapes_used, kept, held_back, by_shape = Counter(), Counter(), [], [], 0

    def keep(expression, rationale, types_used, shape):
        kept.append((expression, rationale))
        used.update(types_used)
        shapes_used[shape] += 1

    for expression, rationale in ideas:
        if len(kept) == n:
            break
        types_used = expression_types(expression, type_map)
        shape = skeleton(expression)
        if any(used[t] >= limit(t) for t in types_used):
            held_back.append((expression, rationale, types_used, shape))
            continue
        if shape in overused_shapes and shapes_used[shape] >= MAX_IDEAS_PER_CROWDED_SHAPE:
            by_shape += 1
            continue
        keep(expression, rationale, types_used, shape)

    let_in = 0
    for expression, rationale, types_used, shape in held_back:
        if len(kept) == n:
            break
        if any(used[t] >= limit(t, 2) for t in types_used):
            continue
        if shape in overused_shapes and shapes_used[shape] >= MAX_IDEAS_PER_CROWDED_SHAPE:
            continue
        keep(expression, rationale, types_used, shape)
        let_in += 1

    if held_back or by_shape:
        logger.info("Variety: skipped %d idea(s) leaning on data types already used enough this round%s and %d "
                    "repeating an overused formula shape.", len(held_back) - let_in,
                    f" ({let_in} more let in to fill the round)" if let_in else "", by_shape)
    if kept:
        logger.info("This round: %d idea(s) in %d different formula shape(s).", len(kept), len(shapes_used))
    if used:
        logger.info("This round's data types: %s", ", ".join(f"{t} x{c}" for t, c in used.most_common()))
    return kept


def build_feedback(last_round_rows=None) -> str:
    """What the generator sees each round: your best results across ALL runs so far, so a new run
    builds on what already worked instead of starting from scratch, plus how the last round went."""
    history = completed_history()
    parts = []
    if history:
        best = sorted(history, key=lambda r: as_float(r["fitness"]), reverse=True)[:MEMORY_TOP_N]
        lines = [f"Your best results so far, out of {len(history)} simulated alphas across all runs "
                 f"(best fitness first). Learn from what works here - the operators, lookback windows, "
                 f"grouping and decay that score well, and what's holding each one back (usually fitness or "
                 f"turnover). But don't keep reusing the same data: a new alpha on the same data as one "
                 f"marked [submitted] or [queued] usually comes back too similar to submit."]
        for r in best:
            lines.append(f"- {r['expression']} -> sharpe={r['sharpe']} fitness={r['fitness']} "
                         f"turnover={r['turnover']}{turnover_note(r)}{returns_note(r)}{pool_tag(r)}{big_stocks_tag(r)}")
        parts.append("\n".join(lines))
    if last_round_rows:
        parts.append(summarize_round(last_round_rows))
    return "\n\n".join(parts) if parts else "This is round 1 - no prior results yet."


CHECKS_EXPLAINED = (
    "BRAIN's checks, so ideas don't walk into them (feedback lists the ones each result didn't pass): "
    "LOW_SUB_UNIVERSE_SHARPE - it must also work among the largest, most liquid stocks, not only in small ones; "
    "CONCENTRATED_WEIGHT - too few stocks get a position, usually sparse data (ts_backfill helps); "
    "LOW_SHARPE / LOW_FITNESS - under BRAIN's bar; HIGH_TURNOVER / LOW_TURNOVER - trades too much or too little; "
    "SELF_CORRELATION - too similar to an alpha already submitted; the units check - adds or subtracts things "
    "measured in different units, such as a plain number added to a data field. So never pad a division with a "
    "tiny number like add(x, 0.0001) or x + 0.0001 - BRAIN copes with dividing by zero - and only add or subtract "
    "fields measured in the same units."
)


def generate_ideas(client: genai.Client, model: str, operators, fields, settings, feedback_text, avoid: set, n: int):
    op_list = "\n".join(f"- {o['name']}: {o['usage']}" for o in operators)
    field_list = "\n".join(f"- {f['name']}: {f['description']}" for f in fields)
    avoid_list = "\n".join(f"- {e}" for e in list(avoid)[:50]) or "(none yet)"

    prompt = (
        f"You are proposing WorldQuant BRAIN FASTEXPR alpha expressions for "
        f"region={settings['region']} universe={settings['universe']} delay={settings['delay']}.\n\n"
        f"ONLY use these operators, exactly as named (do not invent others):\n{op_list}\n\n"
        f"ONLY use these data fields, exactly as named (do not invent others):\n{field_list}\n\n"
        f"BRAIN only accepts an alpha with Sharpe >= {SUBMIT_MIN_SHARPE}, fitness >= {SUBMIT_MIN_FITNESS} "
        f"and turnover between {TURNOVER_MIN} and {TURNOVER_MAX} (1% to 70% of the book traded per day). "
        f"Only just clearing that bar isn't enough to rank well, though: aim for {strong_note()}, and favour "
        f"steady signals over ones that only work in one or two years. "
        f"High turnover also drags fitness down, so favour smoother signals: longer lookback windows, "
        f"ts_decay_linear, or hump. Ideas that scored a strongly negative Sharpe are re-tested reversed "
        f"automatically, and promising ideas that trade too much are re-tested smoothed with "
        f"ts_decay_linear, so don't spend ideas on negating or smoothing past results.\n\n"
        f"{CHECKS_EXPLAINED}\n\n"
        f"{feedback_text}\n\n"
        f"Propose {n} NEW candidate expressions, genuinely different in shape from each "
        f"other (not the same template with one field swapped). Each needs a one-sentence "
        f"economic rationale for why it might predict returns.\n\n"
        f"Do not repeat any of these already-tried expressions:\n{avoid_list}"
    )

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AlphaIdeas,
            # We give the model no tools, so the SDK's automatic function calling has nothing
            # to do - turning it off just removes the "AFC is enabled" log noise.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )

    if not resp.text:
        raise ValueError("Model returned an empty response - possibly blocked by a safety filter.")

    data = json.loads(resp.text)
    return [(idea["expression"], idea["rationale"]) for idea in data["ideas"]]


def is_transient(exc: Exception) -> bool:
    """A busy (503) or rate-limited (429) answer can clear up on its own; a bad key or a
    malformed request can't. google-genai errors carry the HTTP status as `.code`."""
    if isinstance(exc, (httpx.TransportError, requests.RequestException)):
        return True  # dropped or timed-out connection
    return getattr(exc, "code", None) in TRANSIENT_STATUS_CODES


def short_error(exc: Exception) -> str:
    code, status = getattr(exc, "code", None), getattr(exc, "status", None)
    return f"{code} {status}" if code else f"{type(exc).__name__}: {str(exc)[:150]}"


def generate_with_fallback(client: genai.Client, operators, fields, settings, feedback_text, avoid: set, n: int):
    """Main model first, retried with backoff while it's busy or rate-limited. Then the
    fallback model once. Raises the last error only if both fail."""
    last_exc = None
    attempts = len(RETRY_DELAYS_SECONDS) + 1
    for attempt in range(1, attempts + 1):
        try:
            return generate_ideas(client, GEMINI_MODEL, operators, fields, settings, feedback_text, avoid, n)
        except Exception as exc:
            last_exc = exc
            if not is_transient(exc) or attempt == attempts:
                break  # a non-transient error won't fix itself by waiting
            delay = RETRY_DELAYS_SECONDS[attempt - 1]
            logger.warning("%s unavailable (%s) - waiting %ds, then retry %d of %d.",
                           GEMINI_MODEL, short_error(exc), delay, attempt, len(RETRY_DELAYS_SECONDS))
            time.sleep(delay)

    if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL != GEMINI_MODEL:
        logger.warning("%s still failing (%s) - trying %s once.",
                       GEMINI_MODEL, short_error(last_exc), GEMINI_FALLBACK_MODEL)
        try:
            return generate_ideas(client, GEMINI_FALLBACK_MODEL, operators, fields, settings, feedback_text, avoid, n)
        except Exception as exc:
            last_exc = exc

    raise last_exc


def wait_seconds(resp, default=2.0) -> float:
    """BRAIN sends Retry-After as a float string ("5.0"); keep the wait between 0.5 and 30 seconds."""
    try:
        return min(max(float(resp.headers.get("Retry-After")), 0.5), 30.0)
    except (TypeError, ValueError):
        return default


def still_working(resp) -> bool:
    """BRAIN's 'not done yet' signals: a 303 back to the same address, or a Retry-After header."""
    return resp.status_code == 303 or resp.headers.get("Retry-After") is not None


def failing_reason(resp) -> str:
    """Names of the checks BRAIN marked FAIL in a reply, or "" if none are listed."""
    try:
        data = resp.json()
    except ValueError:
        return ""
    checks = []
    if isinstance(data, dict):
        checks = (data.get("is") or {}).get("checks") or data.get("checks") or []
    failed = [str(c.get("name")) for c in checks if isinstance(c, dict) and c.get("result") == "FAIL"]
    return ("failed " + ", ".join(failed)) if failed else ""


def self_correlation_max(session, alpha_id: str):
    """BRAIN's free pre-submit check: the highest correlation between this alpha and the ones you
    already submitted. Returns None if BRAIN didn't give a usable number in time."""
    url = CORRELATION_URL.format(alpha_id=alpha_id)
    deadline = time.time() + CORR_POLL_MAX_SECONDS
    while time.time() < deadline:
        try:
            resp = session.get(url, timeout=30)
        except requests.RequestException:
            return None
        if still_working(resp) and (resp.status_code == 503 or not resp.text.strip()):
            time.sleep(wait_seconds(resp))
            continue
        if resp.status_code != 200:
            return None
        try:
            value = resp.json().get("max")
            return float(value) if value is not None else None
        except (ValueError, TypeError, AttributeError):
            return None
    return None


def submit_alpha(session, alpha_id: str):
    """Submit, then poll until BRAIN gives its verdict. Returns (state, detail), state being one of
    YES, REJECTED, VERDICT_PENDING or FAILED. Following community clients that recorded BRAIN's
    live traffic: 503 or 303 means queued / still processing, Retry-After means poll again, and
    a 403 lists the failed checks. Anything ambiguous is reported as VERDICT_PENDING, never YES."""
    url = SUBMIT_URL.format(alpha_id=alpha_id)
    try:
        resp = session.post(url, timeout=30, allow_redirects=False)
    except requests.RequestException as exc:
        return "FAILED", f"couldn't reach BRAIN ({type(exc).__name__})"
    if resp.status_code == 403:
        return "REJECTED", failing_reason(resp) or f"403 {resp.text[:150]}"
    if resp.status_code == 429:
        return "FAILED", f"BRAIN is throttling submissions right now ({resp.text[:100]})"
    if resp.status_code not in (200, 201, 202, 303, 503):
        return "FAILED", f"{resp.status_code} {resp.text[:150]}"

    logger.info("BRAIN accepted the submission of %s - waiting for its verdict...", alpha_id)
    deadline = time.time() + SUBMIT_POLL_MAX_SECONDS
    while time.time() < deadline:
        time.sleep(wait_seconds(resp))
        try:
            resp = session.get(url, timeout=30, allow_redirects=False)
        except requests.RequestException:
            continue
        if still_working(resp) or resp.status_code == 429:
            continue
        if resp.status_code == 403:
            return "REJECTED", failing_reason(resp) or f"403 {resp.text[:150]}"
        if resp.status_code == 200 and resp.text.strip():
            reason = failing_reason(resp)
            return ("REJECTED", reason) if reason else ("YES", None)
        if resp.status_code == 200:
            return "VERDICT_PENDING", None  # empty answer with no 'wait' signal: can't tell either way
        return "FAILED", f"{resp.status_code} {resp.text[:150]}"
    return "VERDICT_PENDING", None


def is_active_on_brain(session, alpha_id: str) -> bool:
    """Submitted alphas are listed by BRAIN with status ACTIVE (unsubmitted ones: UNSUBMITTED)."""
    return be.fetch_alpha(session, alpha_id).get("status") == "ACTIVE"


def landed(outcome) -> bool:
    """Whether a 'submitted' value means BRAIN took the alpha - what counts toward today."""
    outcome = str(outcome or "")
    return outcome == "YES" or outcome.startswith("VERDICT_PENDING")


# ---- BRAIN's before-and-after check: what this alpha would do to your merged performance ----

def before_after(session, alpha_id: str):
    """Ask BRAIN's free 'before and after' check, the one its Check Submission screen shows: your merged
    performance - what the leaderboard's IS Score ranks - with and without this alpha. Waits while BRAIN
    says it's still working (Retry-After), for up to BEFORE_AFTER_MAX_SECONDS. Returns (status code, reply),
    or (None, None) if BRAIN couldn't be reached."""
    url = (COMPETITION_BEFORE_AFTER_URL.format(competition=ACTIVE_COMPETITION, alpha_id=alpha_id)
           if ACTIVE_COMPETITION else BEFORE_AFTER_URL.format(alpha_id=alpha_id))
    deadline = time.time() + BEFORE_AFTER_MAX_SECONDS
    while True:
        try:
            resp = session.get(url, timeout=be.REQUEST_TIMEOUT)
        except requests.RequestException:
            return None, None
        if still_working(resp) and time.time() < deadline:
            time.sleep(wait_seconds(resp))
            continue
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, (resp.text or "")[:500]


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def before_after_summary(reply) -> str:
    """'score 5627 -> 5710 (+83); sharpe 2.1 -> 2.14; ...' from whatever BRAIN sent, or the reply's field
    names if it isn't laid out as before/after pairs - the full reply is kept in BEFORE_AFTER_FILE either way."""
    if not isinstance(reply, dict):
        return f"reply saved ({type(reply).__name__})"
    parts = []
    score = reply.get("score")
    if isinstance(score, dict):
        before, after = _number(score.get("before")), _number(score.get("after"))
        if before is not None and after is not None:
            parts.append(f"score {before:g} -> {after:g} ({after - before:+g})")
    stats = reply.get("stats")
    if isinstance(stats, dict) and isinstance(stats.get("before"), dict) and isinstance(stats.get("after"), dict):
        for key in ("sharpe", "fitness", "returns", "turnover", "drawdown", "margin"):
            before, after = _number(stats["before"].get(key)), _number(stats["after"].get(key))
            if before is not None and after is not None:
                parts.append(f"{key} {before:.3g} -> {after:.3g}")
    if parts:
        return "; ".join(parts)
    return "reply saved - its fields: " + (", ".join(sorted(str(k) for k in reply)[:12]) or "none")


BEFORE_AFTER_REFUSED = False  # BRAIN said this account may not use the check - don't ask again this run
ACTIVE_COMPETITION = ""       # the competition the before/after check is asked for this run ("": your account's own)


def note_competitions(session):
    """Log the competitions BRAIN lists for you - with your standing there, when BRAIN includes it - and pick the
    one BRAIN's before/after check is asked for: BRAIN_COMPETITION, else the only one you're in. Read-only, and
    never stops the run."""
    global ACTIVE_COMPETITION
    ACTIVE_COMPETITION = BRAIN_COMPETITION
    try:
        resp = session.get(COMPETITIONS_URL, timeout=be.REQUEST_TIMEOUT)
        data = resp.json() if resp.status_code == 200 else None
        found = data.get("results") if isinstance(data, dict) else None
        if not isinstance(found, list):
            logger.info("Couldn't read your BRAIN competitions (%s) - carrying on.", resp.status_code)
            return
        ids = []
        for comp in found:
            if not isinstance(comp, dict) or not comp.get("id"):
                continue
            ids.append(str(comp["id"]))
            board = comp.get("leaderboard") if isinstance(comp.get("leaderboard"), dict) else {}
            standing = ", ".join(f"{key} {board[key]}" for key in ("rank", "score", "alphas")
                                 if board.get(key) not in (None, ""))
            logger.info("Competition %s (%s)%s.", comp["id"], comp.get("name") or "no name",
                        f" - you: {standing}" if standing else "")
        if not ACTIVE_COMPETITION and len(ids) == 1:
            ACTIVE_COMPETITION = ids[0]
        if ACTIVE_COMPETITION:
            logger.info("BRAIN's before/after check will be asked for competition %s (recorded only).",
                        ACTIVE_COMPETITION)
        elif len(ids) > 1:
            logger.info("Set BRAIN_COMPETITION in .env to one of %s to record BRAIN's before/after check for it.",
                        ", ".join(ids))
    except Exception as exc:  # a nice-to-have: whatever went wrong, the run goes on
        logger.info("Couldn't read your BRAIN competitions (%s) - carrying on.", type(exc).__name__)


def record_before_after(session, alpha_id: str, when: str):
    """Record BRAIN's before-and-after check for this alpha: a one-line summary in the log, the full reply
    appended to BEFORE_AFTER_FILE. Only recorded for now - nothing is decided on it. Never stops the run.
    If BRAIN refuses it (403: not open to this account), the rest of the run doesn't ask again."""
    global BEFORE_AFTER_REFUSED
    if BEFORE_AFTER_REFUSED:
        return None
    try:
        status, reply = before_after(session, alpha_id)
        if status is None:
            logger.info("  Couldn't reach BRAIN for the before/after check on %s - carrying on.", alpha_id)
            return None
        Path(BEFORE_AFTER_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(BEFORE_AFTER_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "alpha_id": alpha_id, "when": when,
                                "competition": ACTIVE_COMPETITION or None, "status": status, "reply": reply},
                               ensure_ascii=False, default=str) + "\n")
        if status == 403:
            BEFORE_AFTER_REFUSED = True
            detail = reply.get("detail") if isinstance(reply, dict) else reply
            logger.info("  BRAIN's before/after check isn't open to your account (%s) - not asking again this run.",
                        str(detail)[:150])
            return None
        if status != 200:
            logger.info("  BRAIN's before/after check on %s answered %s: %s", alpha_id, status, str(reply)[:150])
            return None
        logger.info("  Before/after check on %s (%s): %s", alpha_id, when, before_after_summary(reply))
        return reply
    except Exception as exc:
        logger.warning("  The before/after check on %s hit a problem (%s: %s) - carrying on.",
                       alpha_id, type(exc).__name__, str(exc)[:100])
        return None


_UNCHECKED = object()  # try_submit's default: no self-correlation reading taken yet


def too_similar(corr) -> bool:
    return corr is not None and corr >= MAX_SELF_CORRELATION


def daytime_limit() -> float:
    return min(DAYTIME_MAX_SELF_CORRELATION, MAX_SELF_CORRELATION)


def end_of_day_pick() -> bool:
    """Whether the submission being decided now is the end-of-day pick: today's first, once the run has settled
    (no strong alpha went in during the usual rounds, or BRAIN's day is about to end). That one keeps the streak."""
    return SETTLING and submissions_today() == 0


def similar_for_now(corr) -> bool:
    """Under BRAIN's limit, but not under DAYTIME_MAX_SELF_CORRELATION, which every submission except the end-of-day
    pick must be - or unreadable while that limit applies, since then it can't be shown to be under it. Such an
    alpha is held rather than submitted now: it can still be an end-of-day pick."""
    if too_similar(corr) or daytime_limit() >= MAX_SELF_CORRELATION or end_of_day_pick():
        return False
    return corr is None or corr >= daytime_limit()


def daytime_note() -> str:
    """' with self-correlation under 0.5' while the stricter limit is on, '' when it's off."""
    limit = daytime_limit()
    return f" with self-correlation under {limit:g}" if limit < MAX_SELF_CORRELATION else ""


# ---- Portfolio check: what an alpha would do to your submitted alphas combined (the leaderboard's IS Score) ----

PortfolioEffect = namedtuple("PortfolioEffect", "before after corr days alphas")


def daily_pnl(data) -> dict:
    """{date: that day's PnL} from BRAIN's PnL recordset - {"schema": {"properties": [...]}, "records": [[date,
    pnl], ...]} - whose pnl column is the running total (it's what the PnL chart on an alpha's page draws)."""
    schema = data.get("schema") if isinstance(data, dict) else None
    props = schema.get("properties") if isinstance(schema, dict) else None
    names = ([str(p.get("name", "")).lower() if isinstance(p, dict) else str(p).lower() for p in props]
             if isinstance(props, list) else [str(k).lower() for k in props] if isinstance(props, dict) else [])
    at_date = names.index("date") if "date" in names else 0
    at_pnl = names.index("pnl") if "pnl" in names else 1
    rows = []
    for rec in (data.get("records") or []) if isinstance(data, dict) else []:
        if isinstance(rec, (list, tuple)) and len(rec) > max(at_date, at_pnl):
            total = _number(rec[at_pnl])
            if rec[at_date] and total is not None:
                rows.append((str(rec[at_date]), total))
    rows.sort()
    return {day: total - prev for (_, prev), (day, total) in zip(rows, rows[1:])}


def fetch_daily_pnl(session, alpha_id):
    """An alpha's daily PnL from BRAIN, or None if BRAIN didn't send it (after waiting while it's prepared)."""
    url = PNL_URL.format(alpha_id=alpha_id)
    deadline = time.time() + PNL_MAX_SECONDS
    try:
        while True:
            resp = be.with_retries(lambda: session.get(url, timeout=be.REQUEST_TIMEOUT), "reading an alpha's PnL")
            if be.looks_like_expired_session(resp):
                return None
            busy = resp.status_code == 429 or (still_working(resp) and not resp.text.strip())
            if busy and time.time() < deadline:
                time.sleep(wait_seconds(resp))
                continue
            if resp.status_code != 200 or not resp.text.strip():
                return None
            return daily_pnl(resp.json()) or None
    except (requests.RequestException, ValueError):
        return None


def submitted_alpha_ids(session):
    """(ids of your submitted alphas, where they came from): BRAIN's list of your alphas, status ACTIVE - or, if that
    can't be read, the alphas the results file shows going in."""
    ids, offset = [], 0
    try:
        for _ in range(MAX_ALPHA_LIST_PAGES):
            params = {"status": "ACTIVE", "limit": ALPHA_LIST_PAGE, "offset": offset}
            resp = be.with_retries(lambda: session.get(USER_ALPHAS_URL, params=params, timeout=be.REQUEST_TIMEOUT),
                                   "listing your submitted alphas")
            data = resp.json() if resp.status_code == 200 else None
            batch = data.get("results") if isinstance(data, dict) else None
            if not isinstance(batch, list):
                raise ValueError(f"BRAIN answered {resp.status_code}")
            ids += [str(a["id"]) for a in batch if isinstance(a, dict) and a.get("id") and a.get("status") == "ACTIVE"]
            if len(batch) < ALPHA_LIST_PAGE:
                return list(dict.fromkeys(ids)), "BRAIN"
            offset += ALPHA_LIST_PAGE
        logger.warning("Listed only the first %d of your alphas on BRAIN - any submitted alpha past those is left "
                       "out of the portfolio check.", MAX_ALPHA_LIST_PAGES * ALPHA_LIST_PAGE)
        return list(dict.fromkeys(ids)), "BRAIN"
    except (requests.RequestException, ValueError) as exc:
        logger.info("Couldn't list your submitted alphas on BRAIN (%s) - using the results file instead.",
                    str(exc)[:80])
    found = []
    if Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE, encoding="utf-8", newline="") as f:
            found = [r["alpha_id"].strip() for r in csv.DictReader(f)
                     if (r.get("alpha_id") or "").strip() and landed(r.get("submitted"))]
    return list(dict.fromkeys(found)), "the results file"


def annual_sharpe(daily) -> float | None:
    if len(daily) < 2:
        return None
    spread = statistics.pstdev(daily)
    return statistics.fmean(daily) / spread * math.sqrt(252) if spread > 0 else None


def shared_days(all_series) -> list:
    days = set(all_series[0]) if all_series else set()
    for series in all_series[1:]:
        days = days.intersection(series)
    return sorted(days)


def combined_sharpe(all_series):
    """(Sharpe of these daily PnL series combined with equal weights, days they share) - Sharpe None if too few."""
    days = shared_days(all_series)
    if len(days) < PORTFOLIO_MIN_DAYS:
        return None, len(days)
    return annual_sharpe([sum(s[d] for s in all_series) / len(all_series) for d in days]), len(days)


def combined_effect(others, candidate):
    """PortfolioEffect of adding `candidate` (daily PnL) to `others` (the submitted alphas' daily PnL), all equally
    weighted, over the days they all share - None if they share fewer than PORTFOLIO_MIN_DAYS."""
    days = shared_days([candidate, *others])
    if len(days) < PORTFOLIO_MIN_DAYS:
        return None
    new = [candidate[d] for d in days]
    if not others:
        return PortfolioEffect(None, annual_sharpe(new), None, len(days), 0)
    n = len(others)
    before = [sum(series[d] for series in others) / n for d in days]
    after = [(b * n + x) / (n + 1) for b, x in zip(before, new)]
    try:
        corr = statistics.correlation(new, before)
    except statistics.StatisticsError:
        corr = None
    return PortfolioEffect(annual_sharpe(before), annual_sharpe(after), corr, len(days), n)


class Portfolio:
    """Your submitted alphas' daily PnL, read from BRAIN once per run, the first time an alpha is about to go in.
    Alphas that go in during the run are added as they land."""

    def __init__(self):
        self.state = "unread"  # "unread", "ready" or "unavailable" (then the usual rules decide alone)
        self.daily = {}        # alpha_id -> daily PnL, for your submitted alphas
        self.checked = {}      # alpha_id -> daily PnL, for alphas checked this run (kept in case they go in)
        self.blocked = set()   # good alphas this run kept out because they'd lower the combined portfolio

    def ready(self, session) -> bool:
        if self.state == "unread":
            self.state = "unavailable"
            ids, source = submitted_alpha_ids(session)
            daily = {}
            for alpha_id in ids:
                series = fetch_daily_pnl(session, alpha_id) or fetch_daily_pnl(session, alpha_id)
                if not series:
                    logger.warning("Couldn't read the PnL of your submitted alpha %s - the portfolio check is off "
                                   "for this run, so the usual rules decide alone.", alpha_id)
                    return False
                daily[alpha_id] = series
            self.daily, self.state = daily, "ready"
            sharpe, days = combined_sharpe(list(daily.values()))
            logger.info("Portfolio check: %d submitted alpha(s), listed from %s%s. An alpha goes in only if it raises "
                        "their combined Sharpe - the leaderboard's IS Score rates your alphas combined.", len(daily),
                        source, f", combined Sharpe {sharpe:.2f} over {days} shared days" if sharpe is not None else "")
        return self.state == "ready"

    def effect(self, session, alpha_id):
        if not self.ready(session):
            return None
        if alpha_id not in self.checked:
            self.checked[alpha_id] = fetch_daily_pnl(session, alpha_id)
        if not self.checked[alpha_id]:
            return None
        return combined_effect([s for a, s in self.daily.items() if a != alpha_id], self.checked[alpha_id])

    def note_landed(self, alpha_id):
        """An alpha went in: it's part of the portfolio from now on (read again next time if its PnL wasn't read)."""
        series = self.checked.pop(alpha_id, None)
        if self.state == "ready":
            if series:
                self.daily[alpha_id] = series
            else:
                self.state = "unread"


PORTFOLIO = Portfolio()  # a fresh one for every run


def portfolio_effect(session, alpha_id):
    """PORTFOLIO.effect, or None if the check is off or hit a problem (the usual rules then decide alone)."""
    if not PORTFOLIO_CHECK:
        return None
    try:
        return PORTFOLIO.effect(session, alpha_id)
    except Exception as exc:
        logger.warning("The portfolio check on %s hit a problem (%s: %s) - the usual rules decide.",
                       alpha_id, type(exc).__name__, str(exc)[:100])
        return None


def portfolio_verdict(session, alpha_id):
    """(may go in, what the check found). It may go in only if adding it raises your submitted alphas' combined
    Sharpe; the end-of-day pick may lower it by up to END_OF_DAY_MAX_PORTFOLIO_DROP. If the check can't be made
    (it's off, BRAIN didn't send a PnL, too few shared days), the answer is yes and the usual rules decide alone."""
    effect = portfolio_effect(session, alpha_id)
    if effect is None or effect.after is None:
        return True, ("couldn't be made this time, so the usual rules decide" if PORTFOLIO_CHECK else "")
    if effect.before is None:
        return True, f"it would be your first submitted alpha (Sharpe {effect.after:.2f})"
    allowed = END_OF_DAY_MAX_PORTFOLIO_DROP if end_of_day_pick() else 0.0
    change = (effect.after - effect.before) / abs(effect.before) if effect.before else 0.0
    note = (f"your {effect.alphas} submitted alpha(s) combined have Sharpe {effect.before:.2f}, {effect.after:.2f} "
            f"with it ({change:+.1%})" + (f"; its correlation with them is {effect.corr:.2f}"
                                          if effect.corr is not None else ""))
    if effect.after > effect.before - allowed * abs(effect.before):
        return True, note
    PORTFOLIO.blocked.add(alpha_id)
    return False, note + (f" - more than the {allowed:.0%} drop the end-of-day pick may make" if allowed > 0 else "")


def end_of_day_rule() -> str:
    """How the end-of-day pick chooses, in words, for the log."""
    if not PORTFOLIO_CHECK:
        return "the least similar good alpha goes in"
    if END_OF_DAY_MAX_PORTFOLIO_DROP > 0:
        return (f"the good alpha best for your combined portfolio goes in, if it lowers the combined Sharpe by no more "
                f"than {END_OF_DAY_MAX_PORTFOLIO_DROP:.0%}")
    return "the good alpha that raises your combined Sharpe the most goes in - none, if every one would lower it"


def end_of_day_portfolio_order(effect) -> tuple:
    """Sort key: the alpha that leaves the combined Sharpe highest first; ones the check couldn't score last."""
    known = effect is not None and effect.after is not None
    return (not known, -effect.after if known else 0.0)


def try_submit(session, alpha_id: str, corr=_UNCHECKED) -> str:
    """Pre-check self-correlation (unless the caller just did), submit, wait for the verdict.
    Returns the CSV 'submitted' value."""
    if corr is _UNCHECKED:
        corr = self_correlation_max(session, alpha_id)
    if too_similar(corr):
        logger.info("Alpha %s clears the bar, but its self-correlation is %.2f (limit %.1f) - "
                    "not spending today's submission on it.", alpha_id, corr, MAX_SELF_CORRELATION)
        return f"TOO_CORRELATED (max {corr:.2f})"
    if corr is None:
        logger.warning("Couldn't read %s's self-correlation in advance - submitting anyway; "
                       "BRAIN runs the same check during submission.", alpha_id)
    else:
        logger.info("Self-correlation %.2f - under the %.1f limit.", corr, MAX_SELF_CORRELATION)

    record_before_after(session, alpha_id, "before submitting")
    logger.info("Alpha %s qualifies - submitting.", alpha_id)
    state, detail = submit_alpha(session, alpha_id)
    if state in ("YES", "VERDICT_PENDING"):
        record_submission()  # BRAIN took it, so it counts toward today. A rejection doesn't: the run tries the next one
    if state == "YES":
        try:
            if not is_active_on_brain(session, alpha_id):
                state = "VERDICT_PENDING"  # only claim success once BRAIN itself lists it as submitted
        except requests.RequestException:
            state = "VERDICT_PENDING"  # couldn't confirm - the dashboard can, later
    if state == "YES":
        logger.info("SUCCESSFULLY SUBMITTED Alpha ID: %s", alpha_id)
        return "YES"
    if state == "REJECTED":
        logger.warning("BRAIN rejected the submission of %s: %s", alpha_id, detail)
        return f"REJECTED: {detail}"
    if state == "VERDICT_PENDING":
        logger.warning("BRAIN took %s but hasn't confirmed it as submitted yet - press Check with "
                       "BRAIN in the dashboard later.", alpha_id)
        return "VERDICT_PENDING (check with BRAIN in the dashboard)"
    logger.error("Couldn't submit %s: %s", alpha_id, detail)
    return f"FAILED: {detail}"


def needs_smoothing(sharpe, fitness, turnover) -> bool:
    """A decent Sharpe held back by trading too much. Over BRAIN's 70% turnover limit the alpha is
    rejected outright, so smoothing is tried from Sharpe SMOOTH_MIN_SHARPE_OVER_LIMIT (1.0); otherwise
    from the submission bar, when fitness is short. Fitness divides by turnover (floored at 0.125),
    so decay helps most when turnover is high."""
    if sharpe is None or fitness is None or turnover is None:
        return False
    if turnover > TURNOVER_MAX:
        return sharpe >= SMOOTH_MIN_SHARPE_OVER_LIMIT
    if sharpe < SUBMIT_MIN_SHARPE or turnover <= SMOOTH_MIN_TURNOVER:
        return False
    return fitness < SUBMIT_MIN_FITNESS


def smoothing_could_make_it_strong(row) -> bool:
    """A strong Sharpe whose fitness misses the strong bar because it trades a lot: over
    SMOOTH_TOWARD_STRONG_MIN_TURNOVER, and at BRAIN's 12.5% fitness floor with the same returns its fitness would
    reach STRONG_MIN_FITNESS (fitness grows with the square root of the turnover it sheds). Only for alphas with
    no failed check (repairs handle those) that aren't already submitted, rejected or too similar."""
    sharpe, fitness, turnover = (as_float(row.get(k)) for k in ("sharpe", "fitness", "turnover"))
    if sharpe is None or fitness is None or turnover is None:
        return False
    if sharpe < STRONG_MIN_SHARPE or fitness >= STRONG_MIN_FITNESS or turnover <= SMOOTH_TOWARD_STRONG_MIN_TURNOVER:
        return False
    if row.get("failed_checks"):
        return False
    submitted = str(row.get("submitted") or "NO")
    if landed(submitted) or submitted.startswith(("TOO_CORRELATED", "REJECTED")):
        return False
    return fitness * math.sqrt(turnover / FITNESS_TURNOVER_FLOOR) >= STRONG_MIN_FITNESS


# ---- Padding: add(x, 0.0001) and the like ----

_NUMBER = r"(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _is_padding(text: str) -> bool:
    try:
        value = float(text.strip())
    except ValueError:
        return False
    return 0 < abs(value) <= PADDING_MAX


def strip_padding(expression: str) -> str:
    """The expression without add(x, 0.0001)-style padding - a tiny number added so a division can't hit zero.
    It changes nothing that matters, but a plain number added to a data field trips BRAIN's units check.
    Handles add(x, c), add(c, x), x + c and c + x where c is at most PADDING_MAX; anything else is left alone."""
    expr = expression
    changed = True
    while changed:  # add(...) calls, innermost padding first
        changed = False
        for m in re.finditer(r"\badd\s*\(", _blank_quotes(expr)):
            close = matching_paren(expr, m.end() - 1)
            if close < 0:
                continue
            args = split_top_level(expr[m.end():close])
            kept = [a for a in args if not _is_padding(a)]
            if len(args) == 2 and len(kept) == 1:
                expr = expr[:m.start()] + kept[0].strip() + expr[close + 1:]
                changed = True
                break
    # x + c just before a closing bracket, a comma or the end; c + x just after an opening bracket, a comma or the start
    for pattern in (rf"\s*\+\s*({_NUMBER})(?=\s*(?:\)|,|$))", rf"(?:(?<=[(,])|^)\s*({_NUMBER})\s*\+\s*"):
        for m in reversed(list(re.finditer(pattern, _blank_quotes(expr)))):
            if _is_padding(m.group(1)):
                expr = expr[:m.start()] + expr[m.end():]
    return expr


def padding_note(expression: str) -> str:
    return " without its padding" if strip_padding(expression) != expression else ""


# ---- Auto-repair: strong alphas that fail one of BRAIN's fixable checks ----
# Fixes follow community write-ups of BRAIN's checks (rank / ts_backfill for concentrated weight;
# finer groups / no size tilt for weak sub-universe Sharpe). Each repair is one extra simulation,
# and BRAIN's own result decides whether it worked.

_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
COARSE_GROUPS = ("market", "sector", "industry")
NOT_FIELDS = {"true", "false", "nan", "inf"}


def _blank_quotes(text: str) -> str:
    """Same length, with quoted parts (like range='0, 1, 0.1') blanked out so commas and words inside
    them are ignored."""
    return _QUOTED.sub(lambda m: " " * len(m.group()), text)


def matching_paren(text: str, open_index: int) -> int:
    depth = 0
    plain = _blank_quotes(text)
    for i in range(open_index, len(plain)):
        if plain[i] == "(":
            depth += 1
        elif plain[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def split_top_level(text: str) -> list:
    """Split an argument list at its own commas, not at commas inside nested calls or quotes."""
    plain, parts, depth, start = _blank_quotes(text), [], 0, 0
    for i, ch in enumerate(plain):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def outer_call(expression: str):
    """(name, args) when the whole expression is one call like ts_decay_linear(x, 10), else None."""
    expr = expression.strip()
    m = re.match(r"([A-Za-z_]\w*)\s*\(", expr)
    if not m:
        return None
    close = matching_paren(expr, m.end() - 1)
    if close != len(expr) - 1:
        return None
    return m.group(1), split_top_level(expr[m.end():close])


def _group_calls(expression: str):
    """(start of last argument, end of call) for every group_* call - the last argument is the group."""
    plain = _blank_quotes(expression)
    for m in re.finditer(r"\bgroup_\w+\s*\(", plain):
        close = matching_paren(expression, m.end() - 1)
        if close < 0:
            continue
        args = split_top_level(expression[m.end():close])
        yield close - len(args[-1]), close


def data_fields_in(expression: str) -> list:
    """(start, end, name) of each data-field reference: bare words that aren't operators, keyword
    arguments, quoted text or the group argument of a group_* call."""
    plain = _blank_quotes(expression)
    group_args = [(s, e) for s, e in _group_calls(expression)]
    found = []
    for m in _WORD.finditer(plain):
        after = plain[m.end():].lstrip()
        if after.startswith("(") or (after.startswith("=") and not after.startswith("==")):
            continue
        if m.group().lower() in NOT_FIELDS or plain[max(0, m.start() - 1)].isdigit():
            continue
        if any(s <= m.start() and m.end() <= e and expression[s:e].strip() == m.group() for s, e in group_args):
            continue
        found.append((m.start(), m.end(), m.group()))
    return found


def off_list_fields(expression: str, listed) -> set:
    """Data fields BRAIN offers you (data_fields_cache.json) that this expression uses although your field list
    (`listed`, from operators.json) leaves them out. Basic price/volume fields always count as listed; without the
    cache file nothing can be told apart, so nothing is reported."""
    known = field_datasets()
    return {name for _, _, name in data_fields_in(expression)
            if name in known and name not in listed and name not in BASIC_FIELDS}


def field_exists(name: str) -> bool:
    """Whether BRAIN listed this data field for your account (data_fields_cache.json from fetch_fields.py)."""
    try:
        cache = json.loads(Path(FIELDS_CACHE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return any(f.get("id") == name for f in cache.get("matrix_fields") or [] if isinstance(f, dict))


def _wrap_in_rank(expr):
    call = outer_call(expr)
    if call and call[0] == "rank":
        return None
    return (f"rank({expr.strip()})",
            "rank() spreads the weight evenly across stocks instead of a few big bets")


def _backfill_fields(expr):
    fields = data_fields_in(expr)
    if not fields or "ts_backfill(" in expr:
        return None
    for start, end, name in reversed(fields):
        expr = expr[:start] + f"ts_backfill({name}, {BACKFILL_DAYS})" + expr[end:]
    return (expr.strip(), f"ts_backfill fills gaps in sparse data (up to {BACKFILL_DAYS} days), so more stocks "
                          f"get a position")


def _finer_groups(expr):
    changed = False
    for start, end in sorted(_group_calls(expr), reverse=True):
        arg = expr[start:end]
        if arg.strip() in COARSE_GROUPS:
            expr = expr[:start] + arg[:len(arg) - len(arg.lstrip())] + "subindustry" + expr[end:]
            changed = True
    if not changed:
        return None
    return (expr.strip(), "ranking within subindustries removes the size and sector tilts that weaken it in "
                          "the larger, more liquid stocks")


def _size_neutral(expr):
    if "bucket(rank(cap)" in expr or not field_exists("cap"):
        return None
    return (f"group_neutralize({expr.strip()}, bucket(rank(cap), range='0, 1, 0.1'))",
            "neutralizing against market-cap buckets removes the size tilt that weakens it in the more "
            "liquid stocks")


LIQUIDITY = "rank(ts_mean(volume * close, 20))"  # 0 for the least traded stocks in dollars, 1 for the most


def _liquid_split(expr):
    """The community fix for LOW_SUB_UNIVERSE_SHARPE: decay the liquid and the illiquid part separately - the most
    traded stocks on a faster version of the signal, the rest on a slower one. Keeps the alpha's own smoothing as
    the fast speed (5 days without one) and doubles it for the slow."""
    call = outer_call(expr)
    core, fast = expr.strip(), 5
    if call and call[0] in SMOOTHERS and call[1][0].strip():
        core = call[1][0].strip()
        days = _number(call[1][1]) if call[0] == "ts_decay_linear" and len(call[1]) > 1 else None
        fast = int(days) if days is not None and days >= 1 else fast
    return (f"ts_decay_linear({core}, {fast}) * {LIQUIDITY} + ts_decay_linear({core}, {2 * fast}) * (1 - {LIQUIDITY})",
            "the most traded stocks get a faster version of the signal and the rest a slower one, so it holds up "
            "in the larger, more liquid stocks")


def _less_smoothing(expr):
    call = outer_call(expr)
    if not call or call[0] not in ("ts_decay_linear", "hump") or not call[1][0].strip():
        return None
    return (call[1][0].strip(), f"without the outer {call[0]} it trades enough to clear the 1% minimum")


def needs_sharpening(row) -> bool:
    """Sharpe is over the bar but fitness isn't, while turnover is already under BRAIN's 12.5% fitness
    floor - so the smoothing wrapped around it can't help fitness any more, only dampen returns."""
    sharpe, fitness, turnover = (as_float(row.get(k)) for k in ("sharpe", "fitness", "turnover"))
    if sharpe is None or fitness is None or turnover is None:
        return False
    return (sharpe >= SUBMIT_MIN_SHARPE and fitness < SUBMIT_MIN_FITNESS and turnover < FITNESS_TURNOVER_FLOOR
            and _less_smoothing(row.get("expression") or "") is not None)


REPAIRS = {  # check name -> fixes to try, most promising first
    "CONCENTRATED_WEIGHT": (_wrap_in_rank, _backfill_fields),
    "LOW_SUB_UNIVERSE_SHARPE": (_finer_groups, _size_neutral, _liquid_split),
    "LOW_TURNOVER": (_less_smoothing,),
}


def fixable(failed_checks) -> bool:
    """Whether every one of these failed checks has automatic fixes."""
    failed = list(failed_checks or [])
    return bool(failed) and all(c in REPAIRS for c in failed)


def repair_candidates(expression: str, failed_checks, avoid=frozenset()) -> list:
    """(new expression, why) fixes for these failed checks that haven't been tried (aren't in avoid) - one per
    check first, then the next fix for each, up to MAX_REPAIRS_PER_ALPHA. Built on the expression without its
    padding. Empty if any failed check isn't one we know how to fix, or every fix was tried."""
    failed = list(dict.fromkeys(failed_checks))
    if not fixable(failed):
        return []
    base = strip_padding(expression)
    out, seen = [], {expression.strip(), base.strip()}
    for depth in range(max(len(REPAIRS[c]) for c in failed)):
        for check in failed:
            if depth < len(REPAIRS[check]):
                fix = REPAIRS[check][depth](base)
                if fix and fix[0] not in seen and fix[0] not in avoid:
                    seen.add(fix[0])
                    out.append((fix[0], f"failed {check}; {fix[1]}"))
    return out[:MAX_REPAIRS_PER_ALPHA]


def all_repair_variants(expression: str) -> set:
    """Every repair any fixable check could produce - lets a run skip alphas whose repairs were all tried."""
    base = strip_padding(expression)
    return {fix[0] for fixes in REPAIRS.values() for f in fixes for fix in [f(base)] if fix}


def failed_text(checks, names) -> str:
    """The failed checks' names, each with BRAIN's value and limit when it gave them."""
    by_name = {str(c.get("name")): c for c in checks or [] if isinstance(c, dict)}
    out = []
    for name in names:
        check = by_name.get(name) or {}
        value, limit = _number(check.get("value")), _number(check.get("limit"))
        out.append(name + (f" ({value:.2f} vs limit {limit:.2f})" if value is not None and limit is not None else ""))
    return ", ".join(out)


def check_note(check: dict) -> str:
    """'NAME:RESULT', plus BRAIN's value and limit when it gives them:
    'LOW_SUB_UNIVERSE_SHARPE:FAIL (0.62 vs limit 0.87)'."""
    note = f"{check.get('name')}:{check.get('result')}"
    value, limit = _number(check.get("value")), _number(check.get("limit"))
    return note + (f" ({value:.2f} vs limit {limit:.2f})" if value is not None and limit is not None else "")


# ---- BRAIN's big-stocks check (LOW_SUB_UNIVERSE_SHARPE) ----

def sub_universe(checks):
    """(value, limit, result) of BRAIN's big-stocks check - the alpha's Sharpe among the largest, most liquid
    stocks, and the least BRAIN accepts there (about 0.43x its overall Sharpe) - or (None, None, None)."""
    for check in checks or []:
        if isinstance(check, dict) and check.get("name") == "LOW_SUB_UNIVERSE_SHARPE":
            return _number(check.get("value")), _number(check.get("limit")), check.get("result")
    return None, None, None


def out_of_reach(value, limit) -> bool:
    """A Sharpe among the biggest stocks under SUB_UNIVERSE_HOPELESS_SHARE of BRAIN's limit there: the signal
    barely works there at all, and no fix closes a gap that size."""
    return value is not None and limit is not None and limit > 0 and value < SUB_UNIVERSE_HOPELESS_SHARE * limit


def hopeless_gap(checks) -> str:
    """'LOW_SUB_UNIVERSE_SHARPE (0.10 vs limit 0.68)' when BRAIN failed the big-stocks check out of reach, else ''."""
    value, limit, result = sub_universe(checks)
    if result == "FAIL" and out_of_reach(value, limit):
        return f"LOW_SUB_UNIVERSE_SHARPE ({value:.2f} vs limit {limit:.2f})"
    return ""


def small_stocks_only(row) -> bool:
    """A saved result that failed the big-stocks check: its Sharpe among the biggest stocks under the limit."""
    value, limit = as_float(row.get("sub_universe_sharpe")), as_float(row.get("sub_universe_limit"))
    return value is not None and limit is not None and value < limit


def big_stocks_tag(row) -> str:
    if not small_stocks_only(row):
        return ""
    return (f" [fails the big-stocks check: {as_float(row['sub_universe_sharpe']):.2f} vs limit "
            f"{as_float(row['sub_universe_limit']):.2f} - only works in small stocks]")


def idea_core(expression: str) -> str:
    """The idea under the follow-ups wrapped around it - padding, smoothing, rank(), size buckets and the
    liquid/illiquid split come off - so every version of one idea looks alike."""
    expr = strip_padding(expression).strip()
    while True:
        before = expr
        expr = strip_smoothing(expr)
        if LIQUIDITY in expr and expr.startswith("ts_decay_linear("):
            close = matching_paren(expr, len("ts_decay_linear"))
            if close > 0:
                expr = split_top_level(expr[len("ts_decay_linear("):close])[0].strip()
        call = outer_call(expr)
        if call and call[0] == "rank" and len(call[1]) == 1:
            expr = call[1][0].strip()
        elif call and call[0] == "group_neutralize" and len(call[1]) == 2 and "bucket(rank(cap)" in call[1][1]:
            expr = call[1][0].strip()
        if expr == before:
            return expr


def small_stocks_rule() -> str:
    """For the generator: ideas that failed the big-stocks check, one example of each (most recent first), with
    BRAIN's numbers - so it stops writing new versions of them. '' when there are none."""
    seen, lines = set(), []
    rows = sorted((r for r in completed_history() if small_stocks_only(r)),
                  key=lambda r: r.get("timestamp") or "", reverse=True)
    for r in rows:
        core = idea_core(r["expression"])
        key = (skeleton(core), alpha_kind(core))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {core} (Sharpe {r.get('sharpe')} overall, {as_float(r['sub_universe_sharpe']):.2f} among "
                     f"the biggest stocks - needs {as_float(r['sub_universe_limit']):.2f})")
        if len(lines) == SMALL_STOCKS_EXAMPLES:
            break
    if not lines:
        return ""
    return ("ONLY WORKS IN SMALL STOCKS - don't write new versions of these ideas. Each scored well overall but "
            "failed BRAIN's big-stocks check: among the largest, most liquid stocks its Sharpe was far below what "
            "BRAIN needs. Another window, maturity, group, decay or neutralization of the same idea is still the "
            "same idea, and fails the same way. Look for ideas that also work in large stocks:\n" + "\n".join(lines))


def record_sub_universe(alpha_id: str, value, limit):
    """Fill in an earlier result's big-stocks numbers, read from BRAIN when the near-miss was looked at again, so
    the generator hears about it and later runs needn't ask. Written to a temporary file and swapped in; if that
    isn't possible (the file is open elsewhere), the numbers are simply read again next run."""
    with open(RESULTS_FILE, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if "sub_universe_sharpe" not in fieldnames or "sub_universe_limit" not in fieldnames:
        return
    changed = False
    for r in rows:
        if r.get("alpha_id") == alpha_id and not (r.get("sub_universe_sharpe") or r.get("sub_universe_limit")):
            r["sub_universe_sharpe"], r["sub_universe_limit"] = value, limit
            changed = True
    if not changed:
        return
    tmp = Path(RESULTS_FILE + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, RESULTS_FILE)
            return
        except OSError:
            if attempt < REPLACE_ATTEMPTS - 1:
                time.sleep(1)
    try:
        tmp.unlink()
    except OSError:
        pass


def strong_numbers(row) -> bool:
    sharpe, fitness = as_float(row.get("sharpe")), as_float(row.get("fitness"))
    return (sharpe is not None and fitness is not None
            and sharpe >= SUBMIT_MIN_SHARPE and fitness >= SUBMIT_MIN_FITNESS)


def qualifies(metrics: dict, checks: list) -> bool:
    """Every check must pass, except SELF_CORRELATION while it reads PENDING: BRAIN only works
    that one out during submission, so it shows PENDING on every unsubmitted alpha."""
    if not checks:
        return False
    if (safe_float(metrics.get("sharpe")) < SUBMIT_MIN_SHARPE
            or safe_float(metrics.get("fitness")) < SUBMIT_MIN_FITNESS):
        return False
    for c in checks:
        if c.get("result") == "PASS":
            continue
        if c.get("name") == "SELF_CORRELATION" and c.get("result") == "PENDING":
            continue
        return False
    return True


# ---- Strong alphas: well over BRAIN's bar, and earning more than they lose in their worst stretch ----

def strong_note() -> str:
    note = f"Sharpe {STRONG_MIN_SHARPE:g}+, fitness {STRONG_MIN_FITNESS:g}+"
    if STRONG_MIN_RETURNS_TO_DRAWDOWN == 1:
        note += ", returns above drawdown"
    elif STRONG_MIN_RETURNS_TO_DRAWDOWN > 0:
        note += f", returns over {STRONG_MIN_RETURNS_TO_DRAWDOWN:g}x drawdown"
    return note


def is_strong(sharpe, fitness, returns=None, drawdown=None) -> bool:
    """Sharpe and fitness at or over the STRONG_ bars and - unless STRONG_MIN_RETURNS_TO_DRAWDOWN is 0 - annual
    returns above the worst drawdown (times that ratio). Numbers BRAIN didn't report never count as strong."""
    sharpe, fitness = as_float(sharpe), as_float(fitness)
    if sharpe is None or fitness is None or sharpe < STRONG_MIN_SHARPE or fitness < STRONG_MIN_FITNESS:
        return False
    if STRONG_MIN_RETURNS_TO_DRAWDOWN <= 0:
        return True
    returns, drawdown = as_float(returns), as_float(drawdown)
    if returns is None or drawdown is None:
        return False
    return returns > abs(drawdown) * STRONG_MIN_RETURNS_TO_DRAWDOWN


def _blank(value):
    return "" if value is None else value


def row_is_strong(row) -> bool:
    return is_strong(row.get("sharpe"), row.get("fitness"), row.get("returns"), row.get("drawdown"))


def strength(row) -> tuple:
    """Sort key, best first with reverse=True: strong alphas, then fitness, then Sharpe."""
    return row_is_strong(row), safe_float(row.get("fitness")), safe_float(row.get("sharpe"))


# ---- The results file ----

def results_columns():
    """The results file's own column names - None if it doesn't exist yet or is empty."""
    try:
        with open(RESULTS_FILE, encoding="utf-8", newline="") as f:
            return next(csv.reader(f), None) or None
    except OSError:
        return None


def append_result(row: dict):
    """Add one row, in the file's own columns: if an older file couldn't be given the newer columns yet,
    the row still lines up with its header."""
    columns = results_columns()
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        # extrasaction="ignore": rows carry in-memory extras (failing_checks) the CSV doesn't store
        writer = csv.DictWriter(f, fieldnames=columns or RESULTS_HEADER, extrasaction="ignore")
        if not columns:
            writer.writeheader()
        writer.writerow(row)


def upgrade_results_file():
    """Give a results file from an older version the newer columns (returns, drawdown), once, keeping every
    row. Written to a temporary file and swapped in; if Windows won't allow the swap (the file is open in
    another program), new rows keep the old columns and the next run tries again."""
    columns = results_columns()
    if not columns or all(c in columns for c in RESULTS_HEADER):
        return
    with open(RESULTS_FILE, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    header = columns + [c for c in RESULTS_HEADER if c not in columns]
    tmp = Path(RESULTS_FILE + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, RESULTS_FILE)
            logger.info("Added the %s column(s) to %s.", ", ".join(c for c in header if c not in columns),
                        RESULTS_FILE)
            return
        except OSError:
            if attempt < REPLACE_ATTEMPTS - 1:
                time.sleep(1)
    try:
        tmp.unlink()
    except OSError:
        pass
    logger.warning("Couldn't add the new columns to %s (is it open in another program?) - trying again next run.",
                   RESULTS_FILE)


# ---- Never miss a day: keep good alphas back for the coming days ----

_DATASETS = {"key": None, "map": {}}


def field_datasets() -> dict:
    """{field: BRAIN dataset id} from data_fields_cache.json (written by fetch_fields.py). Read again only
    when the file changes - it can be several megabytes."""
    path = Path(FIELDS_CACHE_FILE)
    try:
        stat = path.stat()
    except OSError:
        return {}
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    if _DATASETS["key"] != key:
        found = {}
        try:
            for f in json.loads(path.read_text(encoding="utf-8")).get("matrix_fields") or []:
                dataset = f.get("dataset") if isinstance(f, dict) else None
                dataset = (dataset.get("id") or dataset.get("name")) if isinstance(dataset, dict) else dataset
                if dataset and f.get("id"):
                    found[str(f["id"])] = str(dataset)
        except (OSError, ValueError, AttributeError):
            found = {}
        _DATASETS.update(key=key, map=found)
    return _DATASETS["map"]


PRICE_VOLUME = "price/volume"


def alpha_kind(expression: str) -> frozenset:
    """What an alpha is built on, to tell near-copies apart: the BRAIN dataset of each data field it uses,
    or - for a field the dataset isn't known for - its name without the numbers (implied_volatility_put_90
    and implied_volatility_put_180 are one kind). Basic price/volume data is an ingredient of most alphas,
    so it only counts when there's nothing else. Two alphas sharing any of these usually come back too
    similar to submit both."""
    datasets = field_datasets()
    basic = {datasets.get(name) for name in BASIC_FIELDS} - {None}
    kinds = set()
    for _, _, name in data_fields_in(expression or ""):
        dataset = datasets.get(name)
        if name in BASIC_FIELDS or dataset in basic:
            continue
        kinds.add(dataset or re.sub(r"_+", "_", re.sub(r"\d+", "", name)).strip("_") or name)
    return frozenset(kinds or {PRICE_VOLUME})


def distinct_kinds(kinds) -> int:
    """How many separate groups these kinds make, joining any two that share something."""
    groups = []
    for kind in kinds:
        merged, apart = set(kind), []
        for group in groups:
            if group & merged:
                merged |= group
            else:
                apart.append(group)
        groups = apart + [merged]
    return len(groups)


class Reserve:
    """Good alphas waiting for a coming day - ones BRAIN confirmed during this run (every check passed and
    its similarity check was under the limit) - and the kinds this run has submitted. Counts kinds, not
    alphas: of two near-copies, only one can ever be submitted, because then the other is too similar."""

    def __init__(self):
        self.waiting = {}   # alpha_id -> expression
        self.landed = []    # kind of each alpha submitted during this run

    def keep(self, alpha_id, expression):
        self.waiting[alpha_id] = expression

    def forget(self, alpha_id):
        self.waiting.pop(alpha_id, None)

    def note_landed(self, alpha_id, expression):
        self.forget(alpha_id)
        self.landed.append(alpha_kind(expression))

    def usable(self, besides=None) -> list:
        """Waiting alphas still likely to be submittable if `besides` went in now: anything sharing a kind
        with it, or with what this run already submitted, is left out until BRAIN re-checks it next run."""
        spent = self.landed + ([alpha_kind(besides)] if besides else [])
        return [e for e in self.waiting.values() if not any(alpha_kind(e) & kind for kind in spent)]

    def kinds(self, besides=None) -> int:
        return distinct_kinds(alpha_kind(e) for e in self.usable(besides))


RESERVE = Reserve()  # a fresh one for every run
SETTLING = False      # True once this run has given up waiting for a strong alpha today (settle_for_today)


def may_submit(expression, strong=True) -> bool:
    """Today's first submission: a strong alpha straight away; one that only clears BRAIN's bar once the run
    settles (no strong one by the end of the usual rounds) - the leaderboard counts days. More, up to
    MAX_SUBMISSIONS_PER_DAY, only for strong alphas, and only while at least RESERVE_TARGET other kinds of
    good alpha would still be waiting afterwards, so a day that finds nothing new can still submit."""
    done = submissions_today()
    if done >= MAX_SUBMISSIONS_PER_DAY:
        return False
    if done == 0:
        return strong or SETTLING
    return strong and RESERVE.kinds(besides=expression) >= RESERVE_TARGET


def why_held(expression, strong=True) -> str:
    done = submissions_today()
    if done >= MAX_SUBMISSIONS_PER_DAY:
        return f"today's submission limit ({MAX_SUBMISSIONS_PER_DAY}) is used"
    if not strong:
        if done == 0:
            return f"it isn't strong ({strong_note()})"
        return f"today's submission is in, and only strong alphas ({strong_note()}) go in as an extra"
    left = RESERVE.kinds(besides=expression)
    return (f"today's submission is in, and another now would leave only {left} other kind(s) of good alpha "
            f"for the coming days (keeping {RESERVE_TARGET})")


def settle_for_today(session):
    """No strong alpha has gone in today: from now on any good one that clears BRAIN's bar may, so the day
    still counts - the end-of-day pick, held to BRAIN's own similarity limit only. Submits the least similar
    held alpha straight away, if there's one."""
    global SETTLING
    if MAX_SUBMISSIONS_PER_DAY < 1 or submissions_today() > 0:
        return
    if not SETTLING:
        SETTLING = True
        logger.info("No strong alpha has gone in today - from now on one that only clears BRAIN's bar (Sharpe %s, "
                    "fitness %s) may go in too, so the day can still count. End-of-day pick: %s (held ones first, "
                    "then the next good one found).", SUBMIT_MIN_SHARPE, SUBMIT_MIN_FITNESS, end_of_day_rule())
    submit_held(session)


def reserve_note() -> str:
    usable = RESERVE.usable()
    note = (f"Reserve for the coming days: {len(usable)} good alpha(s) that BRAIN confirmed this run, "
            f"{distinct_kinds(alpha_kind(e) for e in usable)} different kind(s)")
    aside = len(RESERVE.waiting) - len(usable)
    if aside:
        note += (f" - plus {aside} on the same data as today's submission, not counted until BRAIN "
                 f"re-checks them next run")
    return note + "."


# ---- Held alphas: cleared the bar on a day the submission slot was already used ----

def held_candidates() -> list:
    """Held alphas still waiting, best first: strong ones, then by fitness (Sharpe breaks ties). An alpha
    that already has an outcome logged in another row (mark_held's fallback) is left out."""
    if not Path(RESULTS_FILE).exists():
        return []
    with open(RESULTS_FILE, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    held, decided = {}, set()
    for r in rows:
        alpha_id = (r.get("alpha_id") or "").strip()
        submitted = (r.get("submitted") or "").strip()
        if not alpha_id:
            continue
        if submitted.startswith(HELD_PREFIX):
            held.setdefault(alpha_id, r)
        elif submitted not in ("", "NO"):
            decided.add(alpha_id)
    waiting = [r for alpha_id, r in held.items() if alpha_id not in decided]
    return sorted(waiting, key=strength, reverse=True)


def end_of_day_order(item) -> tuple:
    """Sort key for the end-of-day pick, from a (row, self-correlation, latest numbers) entry: the least similar
    to your submitted alphas first, an unreadable reading last, the stronger alpha first between equals."""
    _, corr, latest = item
    strong, fitness, sharpe = strength(latest)
    return corr is None, corr if corr is not None else 0.0, not strong, -fitness, -sharpe


def mark_held(alpha_id: str, outcome: str):
    """Swap a held alpha's 'submitted' value for what happened to it, so the dashboard and later
    runs still see one row per alpha. The CSV is written to a temporary file and swapped in, so a
    crash can't leave it half-written. If Windows blocks the swap (another program has the file
    open), the outcome is appended as a new row instead, so it's never lost."""
    with open(RESULTS_FILE, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or RESULTS_HEADER
        rows = list(reader)
    target = None
    for r in rows:
        if r.get("alpha_id") == alpha_id and (r.get("submitted") or "").startswith(HELD_PREFIX):
            r["submitted"] = outcome
            target = target or r
    if target is None:
        return

    tmp = Path(RESULTS_FILE + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, RESULTS_FILE)
            return
        except OSError:
            if attempt < REPLACE_ATTEMPTS - 1:
                time.sleep(1)
    try:
        tmp.unlink()
    except OSError:
        pass
    logger.warning("Couldn't update %s in place (is it open in another program?) - adding the outcome "
                   "for %s as a new row instead.", RESULTS_FILE, alpha_id)
    append_result(dict(target, round="held", timestamp=time.strftime("%Y-%m-%d %H:%M:%S")))


def submit_held(session):
    """Start-of-run step: if today's slot is free, submit the best held alpha. Returns the first outcome
    BRAIN gave (YES, VERDICT_PENDING..., or REJECTED... if none went in), or None. Never stops the run."""
    try:
        return _submit_held(session)
    except Exception:
        logger.exception("The held-alpha queue hit an unexpected problem - skipping it this run. "
                         "Held alphas stay queued.")
        return None


def _submit_held(session):
    """Re-check every held alpha with BRAIN - including its free similarity check, which grows stricter
    as you submit more - so the queue only keeps alphas that could still be submitted. The ones that pass
    are counted in the reserve. Then, best first (strong ones, then by fitness, on BRAIN's latest numbers): if
    nothing is in for today, a strong one under DAYTIME_MAX_SELF_CORRELATION is submitted - or, once the run
    settles, the least similar one (the end-of-day pick); more only as may_submit() and that limit allow.
    Checking stops after HELD_CHECK_BUDGET_SECONDS; anything not checked yet waits for the next run."""
    queue = held_candidates()
    if not queue:
        return None
    done = submissions_today()
    if done >= MAX_SUBMISSIONS_PER_DAY:
        plan = "Today's limit is used, so they wait for BRAIN's next day (3 AM US Eastern)."
    elif done == 0 and SETTLING:
        plan = f"Nothing is submitted for today yet - end-of-day pick: {end_of_day_rule()}."
    elif done == 0:
        plan = (f"Nothing is submitted for today yet - a strong one ({strong_note()}){daytime_note()} goes in now; "
                f"the others wait until the usual rounds are done, in case a better one turns up.")
    else:
        plan = (f"Today's submission is in - another goes in only if it's strong{daytime_note()} and "
                f"{RESERVE_TARGET}+ other kinds of good alpha would still be waiting for the coming days.")
    logger.info("%d held alpha(s) - checking they're still worth submitting. %s", len(queue), plan)

    # 1. Check them all (best first), so the reserve is known before anything is submitted.
    deadline = time.time() + HELD_CHECK_BUDGET_SECONDS
    good, waiting, dropped = [], 0, 0
    for index, row in enumerate(queue):
        if time.time() > deadline:
            waiting += len(queue) - index
            logger.info("Held-queue check time is up - the rest are checked next run.")
            break
        alpha_id = row["alpha_id"]
        logger.info("Held alpha %s (Sharpe %s, fitness %s): %s", alpha_id, row.get("sharpe"),
                    row.get("fitness"), row.get("expression"))
        try:
            alpha = be.fetch_alpha(session, alpha_id)
        except requests.RequestException as exc:
            logger.warning("Couldn't reach BRAIN to check held alpha %s (%s) - the rest stay queued "
                           "for the next run.", alpha_id, type(exc).__name__)
            waiting += len(queue) - index
            break

        if alpha.get("status") == "ACTIVE":
            logger.info("BRAIN already lists %s as submitted - marking it YES.", alpha_id)
            mark_held(alpha_id, "YES")
            RESERVE.forget(alpha_id)
            continue
        checks = (alpha.get("is") or {}).get("checks") or []
        if not checks:
            logger.warning("BRAIN didn't return %s's checks - leaving it queued.", alpha_id)
            waiting += 1
            continue
        metrics = be.metrics_from_alpha(alpha_id, alpha)
        if not qualifies(metrics, checks):
            failed = [str(c.get("name")) for c in checks if c.get("result") == "FAIL"]
            if (failed or safe_float(metrics.get("sharpe")) < SUBMIT_MIN_SHARPE
                    or safe_float(metrics.get("fitness")) < SUBMIT_MIN_FITNESS):
                reason = ("failed " + ", ".join(failed)) if failed else "below the Sharpe/fitness bar"
                logger.warning("Held alpha %s no longer qualifies on BRAIN (%s) - dropping it from the queue.",
                               alpha_id, reason)
                mark_held(alpha_id, f"NOT_SUBMITTED (no longer qualifies: {reason})")
                RESERVE.forget(alpha_id)
                dropped += 1
            else:
                logger.info("A check on %s isn't settled yet - leaving it queued.", alpha_id)
                waiting += 1
            continue

        corr = self_correlation_max(session, alpha_id)
        if too_similar(corr):
            logger.info("Held alpha %s is now too similar to your submitted alphas (self-correlation %.2f) - "
                        "dropping it from the queue.", alpha_id, corr)
            mark_held(alpha_id, f"TOO_CORRELATED (max {corr:.2f})")
            RESERVE.forget(alpha_id)
            dropped += 1
            continue
        if corr is not None and row.get("submitted") != HELD_PORTFOLIO_LABEL:
            RESERVE.keep(alpha_id, row["expression"])  # confirmed good today - counts toward the reserve
        is_stats = alpha.get("is") or {}
        latest = dict(row, sharpe=metrics.get("sharpe"), fitness=metrics.get("fitness"),
                      returns=is_stats.get("returns"), drawdown=is_stats.get("drawdown"))
        good.append((row, corr, latest))

    # 2. Submit, best first, as far as today allows. The end-of-day pick goes by what each alpha would do to your
    #    combined portfolio (best first), then least similar first.
    good.sort(key=lambda g: strength(g[2]), reverse=True)
    if end_of_day_pick():
        good.sort(key=end_of_day_order)
        if PORTFOLIO_CHECK and good:
            effects = {row["alpha_id"]: portfolio_effect(session, row["alpha_id"]) for row, _, _ in good}
            good.sort(key=lambda g: end_of_day_portfolio_order(effects[g[0]["alpha_id"]]))
    if good and submissions_today() == 0 and not SETTLING and not row_is_strong(good[0][2]):
        logger.info("None of the held alphas is strong (%s) - they wait until the usual rounds are done, in case "
                    "a stronger one turns up.", strong_note())
    first_outcome, submitted, landed_before = None, 0, len(RESERVE.landed)
    for row, corr, latest in good:
        alpha_id, expression = row["alpha_id"], row["expression"]
        if not may_submit(expression, row_is_strong(latest)):
            waiting += 1  # still a good candidate - stays queued
            continue
        if len(RESERVE.landed) > landed_before:
            corr = self_correlation_max(session, alpha_id)  # checked before today's submission went in - ask again
        if similar_for_now(corr):
            reading = (f"its self-correlation is {corr:.2f}" if corr is not None
                       else "its self-correlation couldn't be read")
            logger.info("Held alpha %s could go in, but %s - only alphas%s go in before the end-of-day pick. It "
                        "stays queued: it can still be that pick.", alpha_id, reading, daytime_note())
            waiting += 1
            continue
        if not too_similar(corr):
            allowed, note = portfolio_verdict(session, alpha_id)
            if not allowed:
                logger.info("Held alpha %s stays queued: %s - it would lower your combined portfolio.", alpha_id, note)
                waiting += 1
                continue
            if note:
                logger.info("Portfolio check on held alpha %s: %s.", alpha_id, note)
        outcome = try_submit(session, alpha_id, corr=corr)
        if outcome.startswith("FAILED"):
            logger.warning("Held alpha %s stays queued - the submit didn't go through.", alpha_id)
            waiting += 1
            continue
        mark_held(alpha_id, outcome)
        if landed(outcome):
            RESERVE.note_landed(alpha_id, expression)
            PORTFOLIO.note_landed(alpha_id)
            submitted += 1
            first_outcome = outcome if first_outcome is None or not landed(first_outcome) else first_outcome
        else:  # rejected at submission, or too similar now - not counted toward today, so the next one goes
            RESERVE.forget(alpha_id)
            dropped += 1
            first_outcome = first_outcome or outcome

    logger.info("Held queue: %d submitted, %d dropped (too similar, rejected or no longer qualifying), "
                "%d still waiting.", submitted, dropped, waiting)
    logger.info(reserve_note())
    return first_outcome


BUSY = "BUSY"  # BRAIN is already running as many of your simulations as it allows


def new_row(expression, rationale, round_num) -> dict:
    row = {k: "" for k in RESULTS_HEADER}
    row.update(round=round_num, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
               expression=expression, rationale=rationale, submitted="NO")
    return row


def renew_session(session, email, password, row):
    """One unattended re-login. If BRAIN wants identity verification, logs this idea as STOPPED
    and raises StopLoop - nobody is here to click the link."""
    logger.warning("Session expired mid-run - attempting one re-auth...")
    session.cookies.clear()
    try:
        if log_in_again(session, email, password):
            return
        problem = StopLoop("BRAIN needs re-authentication (likely identity verification) and no one is here to "
                           "complete it.")
    except StopLoop as exc:  # the connection never came back
        problem = exc
    row["status"], row["error"] = "STOPPED", "session expired and could not renew unattended"
    append_result(row)
    raise problem


def is_busy(err) -> bool:
    """BRAIN's 'too many simulations at once' answer (429, or a CONCURRENT_SIMULATION... message)."""
    text = str(err or "").upper()
    return text.startswith("SUBMIT FAILED (429)") or "CONCURRENT_SIMULATION" in text or "SIMULATION_LIMIT" in text


def start_idea(session, row, settings, email, password):
    """Start the simulation. Returns (location, None), (None, BUSY) or (None, error)."""
    location, err = be.submit_simulation(session, row["expression"], settings)
    if err == "EXPIRED_SESSION":
        renew_session(session, email, password, row)
        location, err = be.submit_simulation(session, row["expression"], settings)
    if err and is_busy(err):
        return None, BUSY
    return location, err


def fail(row, error) -> dict:
    row["status"], row["error"] = "FAILED", str(error)[:300]
    append_result(row)
    return row


def hold_similar(session, row, alpha_id, corr):
    """An alpha that could go in now, but whose self-correlation is under BRAIN's limit and not under
    DAYTIME_MAX_SELF_CORRELATION (or couldn't be read): held, so it can still be an end-of-day pick."""
    reading = f"its self-correlation is {corr:.2f}" if corr is not None else "its self-correlation couldn't be read"
    logger.info("Alpha %s qualifies, but %s - only alphas%s go in before the end-of-day pick. Holding it: it can "
                "still be that pick (BRAIN's own limit is %.1f).", alpha_id, reading, daytime_note(),
                MAX_SELF_CORRELATION)
    row["submitted"] = HELD_SIMILAR_LABEL
    if corr is not None:
        RESERVE.keep(alpha_id, row["expression"])
    record_before_after(session, alpha_id, "held as somewhat similar")


def hold_for_portfolio(session, row, alpha_id, note):
    """An alpha that could go in now, but adding it would lower your submitted alphas' combined Sharpe: held (not
    counted in the reserve), and checked again each run, since the portfolio changes as other alphas go in."""
    logger.info("Alpha %s qualifies, but %s - it would lower your combined portfolio, which the leaderboard's IS Score "
                "rates. Holding it in case that changes.", alpha_id, note)
    row["submitted"] = HELD_PORTFOLIO_LABEL
    record_before_after(session, alpha_id, "held: would lower the combined portfolio")


def finish_idea(session, row, alpha_id) -> dict:
    """A simulation finished: read BRAIN's result, log it, and submit the alpha if it qualifies."""
    expression = row["expression"]
    alpha = be.fetch_alpha(session, alpha_id)
    if not (alpha.get("is") or {}):
        row.update(alpha_id=alpha_id, alpha_link=f"https://platform.worldquantbrain.com/alpha/{alpha_id}")
        logger.warning("  %s finished, but BRAIN didn't send its result.", expression)
        return fail(row, "BRAIN finished the simulation but didn't send its result - open its BRAIN page")
    metrics = be.metrics_from_alpha(alpha_id, alpha)
    is_stats = alpha.get("is") or {}
    checks = is_stats.get("checks") or []
    big_value, big_limit, _ = sub_universe(checks)
    row.update(status="COMPLETE", alpha_id=alpha_id, **metrics,
               returns=_blank(is_stats.get("returns")), drawdown=_blank(is_stats.get("drawdown")),
               sub_universe_sharpe=_blank(big_value), sub_universe_limit=_blank(big_limit))
    # Kept in memory for the next round's feedback and the follow-ups; the CSV doesn't store them.
    row["failing_checks"] = ", ".join(check_note(c) for c in checks if c.get("result") != "PASS")
    row["failed_checks"] = [str(c.get("name")) for c in checks if c.get("result") == "FAIL"]
    row["checks"] = checks

    if qualifies(metrics, checks):
        strong = row_is_strong(row)
        if not may_submit(expression, strong):
            # Only hold it if it could actually be submitted later: BRAIN's similarity check is free.
            corr = self_correlation_max(session, alpha_id)
            if too_similar(corr):
                logger.info("Alpha %s clears the bar, but it's too similar to your submitted alphas "
                            "(self-correlation %.2f, limit %.1f) - not holding it.",
                            alpha_id, corr, MAX_SELF_CORRELATION)
                row["submitted"] = f"TOO_CORRELATED (max {corr:.2f})"
            else:
                for_today = not strong and submissions_today() == 0  # may still go in today, if nothing beats it
                logger.warning("Alpha %s qualifies%s, but %s - %s", alpha_id,
                               "" if corr is None else f" (self-correlation {corr:.2f})", why_held(expression, strong),
                               "holding it for now; if no strong alpha turns up in the usual rounds, the "
                               "end-of-day pick chooses from the held ones." if for_today else
                               "holding it for a coming day. Each run starts with the held alphas.")
                row["submitted"] = HELD_FOR_STRONGER_LABEL if for_today else HELD_LABEL
                if corr is not None:
                    RESERVE.keep(alpha_id, expression)
                record_before_after(session, alpha_id, "held for now" if for_today else "held for a coming day")
        else:
            corr = self_correlation_max(session, alpha_id)  # the same free check try_submit makes first
            if similar_for_now(corr):
                hold_similar(session, row, alpha_id, corr)
            else:
                # (one BRAIN would turn away as too similar skips the portfolio check: try_submit logs it as such)
                allowed, note = (True, "") if too_similar(corr) else portfolio_verdict(session, alpha_id)
                if not allowed:
                    hold_for_portfolio(session, row, alpha_id, note)
                else:
                    if note:
                        logger.info("Portfolio check on %s: %s.", alpha_id, note)
                    outcome = try_submit(session, alpha_id, corr=corr)
                    if outcome.startswith("FAILED"):
                        logger.warning("Holding %s so the submit is tried again (before the next extra round, or "
                                       "next run) instead of losing it.", alpha_id)
                        outcome = HELD_RETRY_LABEL
                    elif landed(outcome):
                        RESERVE.note_landed(alpha_id, expression)
                        PORTFOLIO.note_landed(alpha_id)
                    row["submitted"] = outcome

    append_result(row)
    logger.info("  %s -> sharpe=%s fitness=%s turnover=%s checks=%s/%s submitted=%s", expression,
                metrics.get("sharpe"), metrics.get("fitness"), metrics.get("turnover"),
                metrics.get("checks_passed"), metrics.get("checks_total"), row["submitted"])
    return row


def run_round(session, client, operators, fields, settings, email, password, round_num, feedback_text, avoid):
    logger.info("=== Round %d: generating %d ideas ===", round_num, IDEAS_PER_ROUND)
    rejected, dead = rejected_names(), dead_end_fields()
    blocked = rejected | set(dead)
    operators = [o for o in operators if o.get("name") not in rejected]
    fields = [f for f in fields if f.get("name") not in blocked]
    type_map = field_types(fields)
    crowded = crowded_types(type_map)
    overused = crowded_shapes()
    rules = [r for r in (variety_rule(type_map, crowded, IDEAS_PER_ROUND, IDEAS_PER_ROUND + VARIETY_EXTRA_IDEAS),
                         shape_rule(overused, operators),
                         avoid_note(rejected, dead),
                         small_stocks_rule()) if r]
    asked = IDEAS_PER_ROUND + (VARIETY_EXTRA_IDEAS if rules else 0)
    prompt_text = "\n\n".join([feedback_text, *rules])
    try:
        ideas = generate_with_fallback(client, operators, label_fields(fields, type_map), settings,
                                       prompt_text, avoid, asked)
    except Exception as e:
        raise GenerationFailed(short_error(e)) from e

    unpadded = [(strip_padding(e), r) for e, r in ideas]
    padded = sum(1 for (e, _), (u, _) in zip(ideas, unpadded) if e != u)
    if padded:
        logger.info("Took the add(x, 0.0001)-style padding out of %d idea(s) - it only trips BRAIN's units check.",
                    padded)
    ideas = unpadded

    if blocked:
        usable = [(e, r) for e, r in ideas if not uses_any(e, blocked)]
        if len(usable) < len(ideas):
            logger.info("Skipped %d idea(s) using names BRAIN rejected or dead-end fields.", len(ideas) - len(usable))
        ideas = usable
    listed, strays, on_list = {f.get("name") for f in fields}, Counter(), []
    for expression, rationale in ideas:
        off = off_list_fields(expression, listed)
        strays.update(off)
        if not off:
            on_list.append((expression, rationale))
    if strays:
        logger.info("Skipped %d idea(s) using fields that aren't on your field list (%s) - old data the generator "
                    "remembers from past results.", len(ideas) - len(on_list),
                    ", ".join(name for name, _ in strays.most_common(4)))
    ideas = on_list
    if rules:
        fresh_ideas = []
        for expression, rationale in ideas:
            if expression in avoid:
                logger.info("Skipping duplicate: %s", expression)
            else:
                fresh_ideas.append((expression, rationale))
        ideas = pick_varied(fresh_ideas, type_map, crowded, IDEAS_PER_ROUND, overused)

    return process_queue(session, [(e, r, "idea") for e, r in ideas], settings, email, password, round_num, avoid)


def process_queue(session, items, settings, email, password, round_num, avoid) -> list:
    """Test each (expression, rationale, kind), up to PARALLEL_SIMULATIONS at once. A flip or
    smooth goes to the FRONT of the queue, so it starts in the next free slot, right after the
    alpha that triggered it - even a short or interrupted run tests its best leads first.
    Finished simulations are handled one at a time, so submissions never overlap."""
    queue = deque(items)
    running = []  # each: {"row", "kind", "location", "started"}
    rows = []
    counts = {"flips": 0, "smooths": 0, "repairs": 0, "sharpens": 0, "network": 0}
    limit = PARALLEL_SIMULATIONS
    busy_since = None
    last_note = time.time()

    def network_trouble(row, exc):
        counts["network"] += 1
        logger.warning("Couldn't reach BRAIN while testing %s (%s), even after retrying - skipping it. "
                       "Retry it later with --try.", row["expression"], type(exc).__name__)
        rows.append(fail(row, f"Couldn't reach BRAIN ({type(exc).__name__}) - not a verdict on the idea; "
                              f"retry it with --try"))
        if counts["network"] >= MAX_NETWORK_FAILURES:
            raise StopLoop(f"Couldn't reach BRAIN for {counts['network']} ideas in a row - your connection "
                           f"seems to be down. Stopped cleanly; everything so far is saved.")

    def move_clocks(seconds):
        """Time passed without this run (asleep, or waiting for the connection): simulations kept going at BRAIN,
        so their time limits - and BRAIN's busy wait - are moved on by that much."""
        nonlocal busy_since
        for item in running:
            item["started"] += seconds
        if busy_since:
            busy_since += seconds

    def outage(what):
        """The connection itself is gone: wait for it to come back (StopLoop if it doesn't), then carry on."""
        began = time.time()
        if not wait_for_connection(what):
            raise no_connection()
        move_clocks(time.time() - began)
        counts["network"] = 0

    def unexpected(row, exc):  # an unexpected reply shouldn't end the whole run
        logger.error("Unexpected error while testing %s (%s: %s) - skipping it and carrying on.",
                     row["expression"], type(exc).__name__, exc, exc_info=exc)
        rows.append(fail(row, f"Unexpected error: {type(exc).__name__}: {exc}"))

    try:
        while queue or running:
            keep_awake()
            keep_logged_in(session, email, password)
            # 1. Start ideas while there's a free slot.
            while queue and len(running) < limit:
                expression, rationale, kind = queue[0]
                if expression in avoid:
                    queue.popleft()
                    logger.info("Skipping duplicate: %s", expression)
                    continue
                row = new_row(expression, rationale, round_num)
                try:
                    location, err = start_idea(session, row, settings, email, password)
                except StopLoop:
                    raise
                except requests.RequestException as exc:
                    if not brain_reachable():
                        outage("while starting a simulation")  # the idea stays first in line
                        continue
                    queue.popleft()
                    note_tried(avoid, expression)
                    network_trouble(row, exc)
                    continue
                except Exception as exc:
                    queue.popleft()
                    note_tried(avoid, expression)
                    unexpected(row, exc)
                    continue

                if err == BUSY:
                    if running:  # BRAIN's limit is however many we already have going
                        if limit != len(running):
                            limit = len(running)
                            logger.info("BRAIN is running as many of your simulations as it allows - "
                                        "carrying on with %d at a time.", limit)
                        break
                    busy_since = busy_since or time.time()
                    if time.time() - busy_since >= MAX_BUSY_WAIT_SECONDS:
                        queue.popleft()
                        note_tried(avoid, expression)
                        busy_since = None
                        rows.append(fail(row, "BRAIN stayed busy for 10 minutes (simulations running "
                                              "elsewhere?) - retry it with --try"))
                        continue
                    logger.info("BRAIN says it's busy - waiting %ds before starting the next idea.",
                                BUSY_WAIT_SECONDS)
                    move_clocks(nap(BUSY_WAIT_SECONDS))
                    continue

                queue.popleft()
                note_tried(avoid, expression)
                busy_since = None
                counts["network"] = 0
                if err:
                    rows.append(fail(row, err))
                    continue
                running.append({"row": row, "kind": kind, "location": location, "started": time.time()})
                logger.info("Simulating (%d running): %s", len(running), expression)

            if not running:
                continue

            # 2. Check each running simulation once.
            move_clocks(nap(SIM_CHECK_SECONDS))
            if time.time() - last_note >= 60:
                last_note = time.time()
                logger.info("  ... %d simulation(s) running, longest for %ds", len(running),
                            int(time.time() - min(i["started"] for i in running)))
            for item in list(running):
                row = item["row"]
                try:
                    state, info = be.poll_once(session, item["location"])
                except requests.RequestException as exc:
                    if not brain_reachable():
                        outage("while checking on simulations")  # they carry on at BRAIN meanwhile
                        break
                    running.remove(item)
                    network_trouble(row, exc)
                    continue
                except Exception as exc:
                    running.remove(item)
                    unexpected(row, exc)
                    continue

                if state == "RUNNING":
                    if time.time() - item["started"] > be.MAX_POLL_SECONDS:
                        running.remove(item)
                        rows.append(fail(row, f"timed out after {be.MAX_POLL_SECONDS}s"))
                    continue
                running.remove(item)
                if state == "EXPIRED":
                    renew_session(session, email, password, row)  # raises StopLoop if it can't
                    running.append(item)                            # renewed: keep checking it
                    continue
                if state == "ERROR":
                    rows.append(fail(row, info))
                    continue

                counts["network"] = 0
                try:
                    try:
                        row = finish_idea(session, row, info)
                    except be.SessionExpired:
                        renew_session(session, email, password, row)  # raises StopLoop if it can't
                        row = finish_idea(session, row, info)
                except requests.RequestException as exc:
                    if not brain_reachable():
                        running.append(item)  # it finished at BRAIN: its result is read once the connection is back
                        outage("while reading a result")
                        break
                    network_trouble(row, exc)
                    continue
                except Exception as exc:
                    unexpected(row, exc)
                    continue
                rows.append(row)
                queue_followups(session, row, item["kind"], queue, avoid, counts)
    except StopLoop:
        for item in running:  # don't lose track of simulations still going
            item["row"].update(status="STOPPED",
                               error="the run stopped before this simulation finished - retry it with --try")
            append_result(item["row"])
        raise
    return rows


def queue_followups(session, row, kind, queue, avoid, counts):
    """Flip, smooth or repair a finished alpha when that could rescue it. Goes to the front of the queue."""
    if row["status"] != "COMPLETE":
        return
    expression = row["expression"]
    base = strip_padding(expression)  # follow-ups are built without add(x, 0.0001)-style padding
    sharpe, fitness, turnover = (as_float(row.get(k)) for k in ("sharpe", "fitness", "turnover"))

    # Auto-flip: a strongly negative Sharpe means a real signal pointing the wrong way.
    if (kind == "idea" and sharpe is not None and sharpe <= FLIP_SHARPE_AT_OR_BELOW
            and counts["flips"] < MAX_FLIPS_PER_ROUND and not base.startswith("reverse(")):
        flipped = f"reverse({base})"
        if flipped not in avoid:
            counts["flips"] += 1
            logger.info("  Sharpe %.2f points the wrong way - testing the flipped version%s next.", sharpe,
                        padding_note(expression))
            queue.appendleft((flipped, f"Auto-flip of {expression}, which scored Sharpe {sharpe:.2f}; "
                                       f"reversed, it should score about {-sharpe:.2f}.", "flip"))

    # Auto-smooth: the Sharpe is there, but trading too much is holding it back - from BRAIN's bar, or from the
    # strong bar. A sharpened result isn't smoothed back.
    harder = needs_smoothing(sharpe, fitness, turnover)
    if (kind not in ("smooth", "sharpen") and counts["smooths"] < MAX_SMOOTHS_PER_ROUND
            and (harder or smoothing_could_make_it_strong(row))):
        smoothed = f"ts_decay_linear({base}, {SMOOTH_DECAY_DAYS})"
        if smoothed not in avoid:
            counts["smooths"] += 1
            if harder:
                logger.info("  Sharpe %.2f is there, but fitness %.2f at %.0f%% turnover isn't - "
                            "testing a smoothed version%s next.", sharpe, fitness, turnover * 100,
                            padding_note(expression))
                why = f"Sharpe {sharpe:.2f} but fitness {fitness:.2f} at {turnover:.0%} turnover"
            else:
                logger.info("  Sharpe %.2f is strong, but fitness %.2f at %.0f%% turnover misses the strong bar "
                            "(%g) - testing a smoothed version%s next.", sharpe, fitness, turnover * 100,
                            STRONG_MIN_FITNESS, padding_note(expression))
                why = (f"Sharpe {sharpe:.2f} is strong but fitness {fitness:.2f} at {turnover:.0%} turnover misses "
                       f"the strong bar ({STRONG_MIN_FITNESS:g})")
            queue.appendleft((smoothed, f"Auto-smooth of {expression}: {why}; decaying it over "
                                        f"{SMOOTH_DECAY_DAYS} days should cut turnover.", "smooth"))

    # Auto-sharpen: Sharpe is there, fitness isn't, and turnover is already under the 12.5% floor.
    if (kind not in ("smooth", "sharpen", "repair") and counts["sharpens"] < MAX_SHARPENS_PER_ROUND
            and row.get("submitted", "NO") == "NO" and needs_sharpening(row)):
        sharper = (_less_smoothing(base) or (None,))[0]
        if sharper and sharper not in avoid and not too_similar_to_pool(session, row, "sharpening"):
            counts["sharpens"] += 1
            logger.info("  Sharpe %.2f but fitness %.2f at only %.1f%% turnover - testing it without the outer "
                        "smoothing next.", sharpe, fitness, turnover * 100)
            queue.appendleft((sharper, sharpen_reason(expression, sharpe, fitness, turnover), "sharpen"))

    # Auto-repair: strong numbers held back by one of BRAIN's fixable checks.
    if kind != "repair" and row.get("submitted", "NO") == "NO" and strong_numbers(row):
        failed = row.get("failed_checks") or []
        gap = hopeless_gap(row.get("checks"))
        if gap:
            logger.info("  Strong numbers but failed %s - it barely works among the biggest stocks, so no fix can "
                        "get it through; not repairing it.", gap)
        elif not fixable(failed):
            if failed:
                logger.info("  Strong numbers but failed %s - no automatic fix for that check.",
                            failed_text(row.get("checks"), failed))
            elif row.get("failing_checks"):
                logger.info("  Strong numbers but not every check passed (%s).", row["failing_checks"])
        fixes = [] if gap else repair_candidates(expression, failed, avoid)
        room = MAX_REPAIRS_PER_ROUND - counts["repairs"]
        if fixes and room > 0 and worth_repairing(session, row):
            counts["repairs"] += min(room, len(fixes))
            for new_expression, why in reversed(fixes[:room]):  # reversed: appendleft keeps their order
                queue.appendleft((new_expression, f"Auto-repair of {expression}: {why}.", "repair"))


def sharpen_reason(expression, sharpe, fitness, turnover) -> str:
    return (f"Auto-sharpen of {expression}: Sharpe {sharpe:.2f} but fitness {fitness:.2f} at {turnover:.1%} "
            f"turnover - under BRAIN's 12.5% fitness floor, so the outer smoothing only dampens returns.")


def too_similar_to_pool(session, row, purpose: str) -> bool:
    """A follow-up is a close copy of the original, so if the original is already too similar to your
    submitted alphas, the follow-up would be too. BRAIN's similarity check is free; asking first saves
    the simulation."""
    corr = self_correlation_max(session, row["alpha_id"])
    if too_similar(corr):
        logger.info("  %s is too similar to your submitted alphas (self-correlation %.2f) - not %s it.",
                    row["expression"], corr, purpose)
        return True
    return False


def worth_repairing(session, row) -> bool:
    if too_similar_to_pool(session, row, "repairing"):
        return False
    logger.info("  Strong numbers but failed %s - testing a repaired version%s next.",
                failed_text(row.get("checks"), row.get("failed_checks") or []), padding_note(row["expression"]))
    return True


def near_miss_rows() -> list:
    """Earlier alphas with Sharpe and fitness over the bar that weren't submitted - most recent first."""
    if not Path(RESULTS_FILE).exists():
        return []
    with open(RESULTS_FILE, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("status") == "COMPLETE" and r.get("alpha_id") and strong_numbers(r)
                and (r.get("submitted") or "NO").strip() == "NO"]
    unique = {r["alpha_id"]: r for r in rows}  # one row per alpha
    return sorted(unique.values(), key=lambda r: r.get("timestamp") or "", reverse=True)


def sharpen_candidates() -> list:
    """Earlier alphas that needs_sharpening() and weren't submitted - most recent first."""
    rows = [r for r in completed_history() if r.get("alpha_id") and needs_sharpening(r)
            and (r.get("submitted") or "NO").strip() == "NO"]
    unique = {r["alpha_id"]: r for r in rows}
    return sorted(unique.values(), key=lambda r: r.get("timestamp") or "", reverse=True)


def repair_past_near_misses(session, settings, email, password, avoid) -> list:
    """Start-of-run step: re-examine up to MAX_PAST_REPAIRS recent near-misses whose follow-ups haven't
    been tried - repair the ones failing a fixable check (asking BRAIN which check failed), and test the
    slow ones without their outer smoothing. Every near-miss looked at gets a log line saying why."""
    candidates = [("repair", r) for r in near_miss_rows()] + [("sharpen", r) for r in sharpen_candidates()]
    candidates.sort(key=lambda kr: kr[1].get("timestamp") or "", reverse=True)
    items, looked = [], 0
    for kind, row in candidates:
        if looked >= MAX_PAST_REPAIRS:
            break
        expression = row["expression"]
        if kind == "sharpen":
            sharper = (_less_smoothing(strip_padding(expression)) or (None,))[0]
            if not sharper or sharper in avoid:
                continue  # already tried
            looked += 1
            logger.info("Earlier near-miss %s (Sharpe %s, fitness %s, turnover %s):", expression,
                        row.get("sharpe"), row.get("fitness"), row.get("turnover"))
            if too_similar_to_pool(session, row, "sharpening"):
                continue
            logger.info("  Turnover is under BRAIN's 12.5% fitness floor - testing it without the outer smoothing.")
            items.append((sharper, sharpen_reason(expression, *(as_float(row[k]) for k in
                                                                ("sharpe", "fitness", "turnover"))), "sharpen"))
            continue

        known = bool((row.get("sub_universe_sharpe") or "").strip())  # "-" once BRAIN was asked and had none
        if known and out_of_reach(as_float(row.get("sub_universe_sharpe")), as_float(row.get("sub_universe_limit"))):
            continue  # it barely works among the biggest stocks - looked at before, nothing to fix
        if all_repair_variants(expression) <= avoid and known:
            continue  # every possible fix was already tried (its big-stocks numbers may still need filling in)
        looked += 1
        try:
            alpha = be.fetch_alpha(session, row["alpha_id"])
        except requests.RequestException as exc:
            logger.warning("Couldn't reach BRAIN to look at earlier near-misses (%s) - skipping them this run.",
                           type(exc).__name__)
            break
        checks = (alpha.get("is") or {}).get("checks") or []
        row["failed_checks"] = [str(c.get("name")) for c in checks if c.get("result") == "FAIL"]
        row["checks"] = checks
        label = f"Earlier near-miss {expression} (Sharpe {row.get('sharpe')}, fitness {row.get('fitness')})"
        if not checks:
            logger.info("%s: couldn't read BRAIN's record of it - trying again next run.", label)
            continue
        if not known:
            big_value, big_limit, _ = sub_universe(checks)
            record_sub_universe(row["alpha_id"], "-" if big_value is None else big_value,
                                "-" if big_limit is None else big_limit)
        if all_repair_variants(expression) <= avoid:
            continue  # its numbers are filled in now; every possible fix was already tried
        gap = hopeless_gap(checks)
        if gap:
            logger.info("%s fails %s - it barely works among the biggest stocks, so no fix can get it through; "
                        "not repairing it.", label, gap)
            continue
        if not row["failed_checks"]:
            pending = ", ".join(check_note(c) for c in checks
                                if c.get("result") != "PASS" and c.get("name") != "SELF_CORRELATION")
            logger.info("%s: BRAIN lists no failed check%s.", label, f" ({pending})" if pending else "")
            continue
        if not fixable(row["failed_checks"]):
            logger.info("%s fails %s - no automatic fix for that check.", label,
                        failed_text(checks, row["failed_checks"]))
            continue
        fixes = repair_candidates(expression, row["failed_checks"], avoid)
        if not fixes:
            continue  # its fixes were already tried
        logger.info("%s:", label)
        if not worth_repairing(session, row):
            continue
        items.extend((e, f"Auto-repair of {expression}: {why}.", "repair") for e, why in fixes)
    if not items:
        return []
    logger.info("=== Following up earlier near-misses: %d version(s) to test ===", len(items))
    return process_queue(session, items, settings, email, password, "repair", avoid)


def run_try(session, expressions, settings, email, password) -> list:
    """--try: your own expressions through the full pipeline, flips and smooths included.
    Skips anything already simulated successfully; retries ones that failed before."""
    done = {r["expression"] for r in completed_history()}
    items = []
    for expression in (e.strip() for e in expressions):
        if not expression:
            continue
        if expression in done:
            logger.info("Skipping %s - it was already simulated; its result is in the dashboard.", expression)
            continue
        items.append((expression, "Requested with --try", "idea"))
    if not items:
        logger.info("Nothing new to try.")
        return []
    avoid = already_tried() - {e for e, _, _ in items}  # still avoid repeating everything else
    logger.info("Trying %d expression(s) you gave.", len(items))
    return process_queue(session, items, settings, email, password, "try", avoid)


LOCK_FILE = "alpha_loop.lock"


def acquire_run_lock(path=LOCK_FILE):
    """Hold an operating-system lock on a small file for the whole run, so a scheduled run and a manual
    run never overlap - they'd share the results file and the daily submission count. The OS drops the
    lock when the process ends, even after a crash. Returns the open file (keep it open), or None if
    another run holds it."""
    handle = open(path, "a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def add_log_file(log_dir) -> logging.Handler:
    """Also write this run's log to <log_dir>/alpha_loop_YYYY-MM-DD.log (UTF-8, appended), for runs
    nobody is watching, such as the daily scheduled one."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(Path(log_dir) / f"alpha_loop_{time.strftime('%Y-%m-%d')}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate, test and submit WorldQuant BRAIN alphas.")
    parser.add_argument("--try", dest="try_expressions", nargs="+", metavar="EXPRESSION",
                        help="simulate these exact expressions (and submit any that qualify) instead of "
                             "asking Gemini for ideas; wrap each one in double quotes")
    parser.add_argument("--log-dir", metavar="FOLDER",
                        help="also write the log to FOLDER/alpha_loop_YYYY-MM-DD.log (for scheduled runs)")
    parser.add_argument("--backup", action="store_true",
                        help="for a second scheduled run later in the day: if today's submission is already in "
                             "and the reserve is full, stop after re-checking the held alphas")
    parser.add_argument("--catch-up", dest="catch_up", action="store_true",
                        help="for extra scheduled triggers (start-up, reconnecting, every few hours): does nothing "
                             "if today's submission is already in or another run is going; otherwise runs until "
                             "today's submission lands")
    args = parser.parse_args(argv)

    log_file = add_log_file(args.log_dir) if args.log_dir else None
    try:
        if args.catch_up and submissions_today() > 0:
            logger.info("Catch-up run: today's submission is already in - nothing to do.")
            return
        lock = acquire_run_lock()
        if lock is None:
            if args.catch_up:
                logger.info("Catch-up run: another run is going - leaving today to it.")
            else:
                logger.error("Another alpha_loop run is still going (it holds %s) - not starting a second one. "
                             "Try again once it prints Done.", LOCK_FILE)
            return
        try:
            run(args)
        except LoginFailed as e:
            logger.error("Stopping: %s", e)
        except Exception:
            logger.exception("The run crashed - the details below show where.")
            raise
        finally:
            let_sleep()
            lock.close()
    finally:
        if log_file:
            logging.getLogger().removeHandler(log_file)
            log_file.close()


def gemini_client() -> genai.Client:
    """The Gemini client, with a GEMINI_TIMEOUT_SECONDS limit on each request (the library's default is none)."""
    key = os.environ["GEMINI_API_KEY"]
    try:
        return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_SECONDS * 1000))
    except (TypeError, ValueError, AttributeError) as exc:  # an older google-genai without per-request limits
        logger.warning("This google-genai can't put a time limit on Gemini requests (%s) - carrying on without "
                       "one. pip install -U google-genai adds it.", type(exc).__name__)
        return genai.Client(api_key=key)


def run(args):
    global RESERVE, BEFORE_AFTER_REFUSED, SETTLING, OUTAGE_SECONDS, PORTFOLIO
    RESERVE = Reserve()  # the reserve only counts alphas BRAIN confirms during this run
    PORTFOLIO = Portfolio()  # your submitted alphas' PnL is read afresh each run
    BEFORE_AFTER_REFUSED = False
    SETTLING = False
    OUTAGE_SECONDS = 0.0
    email, password = be.get_credentials()
    upgrade_results_file()
    keep_awake()
    if args.try_expressions:
        session = connect(email, password)
        note_competitions(session)
        submit_held(session)
        try:
            rows = run_try(session, args.try_expressions, SETTINGS, email, password)
        except StopLoop as e:
            logger.error("Stopping: %s", e)
            return
        settle_for_today(session)  # no rounds to wait for: if no strong alpha went in, the end-of-day pick
        logger.info("Done: %d simulation(s), including any automatic flips or smooths. Results are in %s.",
                    len(rows), RESULTS_FILE)
        return

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("Set GEMINI_API_KEY in your .env file too (same format as WQ_EMAIL/WQ_PASSWORD).")

    client = gemini_client()

    operators, fields = load_reference()
    if len(operators) <= 6:
        logger.warning(
            "operators.json still has the small starter set (%d operators) - BRAIN's real "
            "list has hundreds. Ideas will be limited and repetitive until you replace it "
            "with your export from platform.worldquantbrain.com/learn/operators.",
            len(operators),
        )

    settings = SETTINGS
    session = connect(email, password)
    note_competitions(session)
    submit_held(session)  # before building feedback, so the generator sees the updated outcome
    catch_up = getattr(args, "catch_up", False)
    if catch_up and submissions_today() > 0:
        logger.info("Catch-up run: a held alpha covered today - nothing more to do.")
        return
    if getattr(args, "backup", False) and submissions_today() > 0:
        spare = RESERVE.kinds()
        if spare >= RESERVE_TARGET:
            logger.info("Backup run: today's submission is in and %d kinds of good alpha are waiting for the "
                        "coming days - nothing more to do.", spare)
            return
        logger.info("Backup run: today's submission is in, but only %d kind(s) of good alpha are waiting "
                    "(aiming for %d) - mining to build the reserve.", spare, RESERVE_TARGET)
    avoid = already_tried()
    try:
        repair_past_near_misses(session, settings, email, password, avoid)
    except StopLoop as e:
        logger.error("Stopping: %s", e)
        return
    except Exception:
        logger.exception("Repairing earlier near-misses hit an unexpected problem - skipping it this run.")
    feedback_text = build_feedback()
    past = len(completed_history())
    if past:
        logger.info("Starting from your best results across %d past simulations.", past)

    logger.info("Starting: up to %d rounds x %d ideas (%d simulations, %d at a time), %ds runtime cap, "
                "%d submission(s)/day max.", MAX_ROUNDS, IDEAS_PER_ROUND, MAX_ROUNDS * IDEAS_PER_ROUND,
                PARALLEL_SIMULATIONS, MAX_RUNTIME_SECONDS, MAX_SUBMISSIONS_PER_DAY)
    if MAX_SUBMISSIONS_PER_DAY > 0:
        logger.info("Strong alphas (%s)%s go in first%s. If none has by the end of the %d rounds, end-of-day pick: %s.",
                    strong_note(), daytime_note(), ", if they raise your combined portfolio" if PORTFOLIO_CHECK else "",
                    MAX_ROUNDS, end_of_day_rule())
        logger.info("Every day counts: with nothing submitted for today after %d rounds, up to %d more rounds "
                    "until something lands.%s", MAX_ROUNDS, MAX_EXTRA_ROUNDS,
                    f" A submission beyond today's first must be strong, with {RESERVE_TARGET}+ other kinds of good "
                    f"alpha left in reserve." if MAX_SUBMISSIONS_PER_DAY > 1 else "")
    type_map = field_types(fields)
    if type_map:
        logger.info("Variety: your %d fields span %d data types (%s).", len(type_map),
                    len(set(type_map.values())), ", ".join(sorted(set(type_map.values()))))
    rejected, dead = rejected_names(), dead_end_fields()
    if rejected or dead:
        logger.info("Leaving out %d name(s) BRAIN rejected (%s) and %d dead-end field(s) (%s).", len(rejected),
                    ", ".join(sorted(rejected)) or "none", len(dead), ", ".join(sorted(dead)) or "none")

    started, paused_before = time.time(), OUTAGE_SECONDS
    rounds_done = 0
    failures_in_a_row = 0
    checked_at = None  # the round count the end-of-rounds step last ran for
    stopped = False    # the login couldn't be renewed - no last call then
    day = brain_day()
    while True:
        keep_awake()
        if time.time() - started - (OUTAGE_SECONDS - paused_before) > MAX_RUNTIME_SECONDS:
            logger.info("Hit MAX_RUNTIME_SECONDS - stopping cleanly.")
            break
        if catch_up and submissions_today() > 0:
            logger.info("Catch-up run: today's submission is in - stopping here.")
            break
        if brain_day() != day:  # the run went past BRAIN's day end: the new day starts strong-first again
            day, SETTLING = brain_day(), False
            logger.info("BRAIN day %s has started.", day)
        if (MAX_SUBMISSIONS_PER_DAY > 0 and not SETTLING and submissions_today() == 0
                and brain_day_ends_in() <= timedelta(minutes=SETTLE_BEFORE_DAY_END_MINUTES)):
            logger.info("BRAIN's day ends in %d minutes and nothing is in for it yet - settling now rather than "
                        "waiting for a strong alpha.", brain_day_ends_in().total_seconds() // 60)
            settle_for_today(session)
        if rounds_done >= MAX_ROUNDS:
            if MAX_SUBMISSIONS_PER_DAY < 1 or submissions_today() > 0:
                break
            if checked_at != rounds_done:  # once per round, not again after a failed generation
                checked_at = rounds_done
                try:
                    keep_logged_in(session, email, password)
                except StopLoop as e:
                    logger.error("Stopping: %s", e)
                    stopped = True
                    break
                settle_for_today(session)  # the end-of-day pick goes in now, strong or not
                if submissions_today() > 0:
                    break
                if MAX_ROUNDS < 1:
                    break
                if rounds_done >= MAX_ROUNDS + MAX_EXTRA_ROUNDS:
                    logger.warning("Still nothing submitted for today after %d extra round(s) - stopping here.",
                                   MAX_EXTRA_ROUNDS)
                    break
                if rounds_done == MAX_ROUNDS:
                    logger.info("Nothing submitted for BRAIN day %s yet - carrying on past MAX_ROUNDS until "
                                "something lands (up to %d more rounds, within the %d-minute runtime cap).",
                                brain_day(), MAX_EXTRA_ROUNDS, MAX_RUNTIME_SECONDS // 60)
        round_num = rounds_done + 1
        try:
            round_rows = run_round(session, client, operators, fields, settings, email, password,
                                    round_num, feedback_text, avoid)
        except GenerationFailed as e:
            if not brain_reachable():  # it's the connection, not Gemini: wait for it, without counting a failure
                if wait_for_connection("while asking Gemini for ideas"):
                    continue
                logger.error("Stopping: %s", no_connection())
                stopped = True
                break
            failures_in_a_row += 1
            if failures_in_a_row >= MAX_GENERATION_FAILURES:
                logger.error(
                    "Idea generation failed %d times in a row (last: %s). Stopping - nothing was "
                    "simulated, so no BRAIN quota was used. Gemini is likely overloaded; try again later.",
                    failures_in_a_row, e)
                break
            logger.warning(
                "Round %d didn't start - idea generation failed (%s). Waiting %ds and trying again; "
                "this does not use up a round.", round_num, e, GENERATION_COOLDOWN_SECONDS)
            nap(GENERATION_COOLDOWN_SECONDS)
            continue
        except StopLoop as e:
            logger.error("Stopping: %s", e)
            stopped = True
            break
        failures_in_a_row = 0
        rounds_done += 1
        feedback_text = build_feedback(round_rows)
        nap(5)

    if not stopped and not SETTLING:
        settle_for_today(session)  # ended before the usual rounds were done (runtime cap, Gemini down)
    extra = max(0, rounds_done - MAX_ROUNDS)
    logger.info("Done: %d round(s) of ideas tested%s. Full history is in %s.", rounds_done,
                f" ({extra} of them extra, to get today's submission in)" if extra else "", RESULTS_FILE)
    if MAX_SUBMISSIONS_PER_DAY > 0 and submissions_today() == 0:
        logger.warning("Nothing was submitted for BRAIN day %s. A later run before its day ends (3 AM US "
                       "Eastern) - such as the backup run - will try again.", brain_day())
        if PORTFOLIO.blocked:
            logger.warning("%d good alpha(s) were kept out because each would have lowered your combined portfolio. "
                           "To keep the daily streak at a small cost, set END_OF_DAY_MAX_PORTFOLIO_DROP in .env "
                           "(0.02 lets the end-of-day pick lower the combined Sharpe by up to 2%%).",
                           len(PORTFOLIO.blocked))
    logger.info(reserve_note())


if __name__ == "__main__":
    main()
