/*
 * burst_ctrl.c — Burst / cool-down duty-cycle controller (M1)
 *
 * The per-decode-step controller that picks a "compute profile" for each
 * step: BURST (cheap/coarse — pyramid L1-only, fewer experts, activation-
 * skip, optional early-exit) or COOLDOWN (precise/anchoring — full L1+L2,
 * all experts). The pyramid PQ format already carries L1 (coarse) + L2
 * (residual); a BURST step reads L1-only, a COOLDOWN step reads L1+L2.
 * Speculative decoding's verify pass is the cool-down anchor.
 *
 * This file owns ONLY the controller itself: the state machine, the
 * accept-rate EMA, the active-profile pointer, and the public attach /
 * set / metrics entry points. The kernels that actually consume a profile
 * (early-exit truncated forward, the L1-only matmul byte path, gate-energy
 * expert selection, runtime KV re-quant) live in their owning files and
 * are filled in by later (M2) agents.
 *
 * INVARIANT: with cfg.enabled == 0 (the default) the controller always
 * returns IB_PROFILE_EXACT, never leaves the exact profile, and the
 * active_skip_thresh_ratio is whatever IB_PQV2_SKIP seeded at attach — so
 * a burst-disabled run is byte-identical to today.
 */

#include "inferbit_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The EXACT profile: full depth, L1+L2, all experts, no skip. A zero-init
 * ib_compute_profile is NOT exact (max_layer 0 would mean "run zero
 * layers"), so the exact profile is spelled out explicitly:
 *   max_layer=-1, precision_tier=0, mome_top_n=-1,
 *   skip_thresh_ratio=0, kv_bits=0, backend_hint=0. */
static const ib_compute_profile g_profile_exact = { -1, 0, -1, 0.0f, 0, 0 };

/* Parse IB_PQV2_SKIP once (if set) into a [0,1) activation-skip ratio.
 * Returns 0.0f when unset/invalid — preserving the no-skip default. The
 * threaded matmul caches this (× xmax) instead of calling getenv per call;
 * seeding it here at attach time reproduces today's env-driven behaviour
 * for the default, burst-disabled run. */
static float ib_burst_env_skip_ratio(void) {
    const char *env = getenv("IB_PQV2_SKIP");
    if (env && env[0]) {
        float ratio = (float)atof(env);
        if (ratio > 0.0f && ratio < 1.0f) return ratio;
    }
    return 0.0f;
}

/* getenv → int with a default. Treats unset/empty as `def`. */
static int ib_env_int(const char *name, int def) {
    const char *e = getenv(name);
    if (!e || !e[0]) return def;
    return atoi(e);
}

/* getenv → float with a default. Treats unset/empty as `def`. */
static float ib_env_float(const char *name, float def) {
    const char *e = getenv(name);
    if (!e || !e[0]) return def;
    return (float)atof(e);
}

/* Build a burst config from IB_BURST* env vars (testability, M2).
 *
 * Returns 1 and fills *out when IB_BURST is set (enabling the duty cycle);
 * returns 0 when IB_BURST is unset/0 (caller leaves the feature disabled, so
 * the load-time default path is byte-identical to today). Env knobs:
 *   IB_BURST=1            enable
 *   IB_BURST_PERIOD=N     cooldown_period (default 8)
 *   IB_BURST_L1ONLY=1     burst.precision_tier=1 (L1-only coarse tier)
 *   IB_BURST_TOPN=k       burst.mome_top_n (default -1 = expert sparsity off)
 *   IB_BURST_SKIP=ratio   burst.skip_thresh_ratio (default 0)
 *   IB_BURST_ACCEPT_FLOOR / IB_BURST_MARGIN_FLOOR (default 0 = period only)
 * The cool-down profile is the EXACT dials (full L1+L2, all-K, no skip,
 * full depth). burst.max_layer stays -1: early-exit is NOT enabled by
 * default (skipped-layer KV holes need the verify backfill = M3). */
static int ib_burst_env_config(ib_burst_config *out) {
    const char *en = getenv("IB_BURST");
    int enabled = (en && en[0] && en[0] != '0');
    memset(out, 0, sizeof(*out));
    out->enabled = enabled ? 1 : 0;
    if (!enabled) return 0;

    out->cooldown_period = ib_env_int("IB_BURST_PERIOD", 8);
    out->accept_floor    = ib_env_float("IB_BURST_ACCEPT_FLOOR", 0.0f);
    out->margin_floor    = ib_env_float("IB_BURST_MARGIN_FLOOR", 0.0f);

    /* BURST profile: cheap/coarse dials from the env. */
    out->burst.max_layer        = ib_env_int("IB_BURST_MAXLAYER", -1); /* early-exit draft depth; -1 = full. Safe under self-spec: draft KV is discarded and the verify rewrites full depth. */
    out->burst.precision_tier   = ib_env_int("IB_BURST_L1ONLY", 0) ? 1 : 0;
    out->burst.mome_top_n       = ib_env_int("IB_BURST_TOPN", -1);
    out->burst.skip_thresh_ratio= ib_env_float("IB_BURST_SKIP", 0.0f);
    out->burst.kv_bits          = 0;
    out->burst.backend_hint     = 0;

    /* COOL-DOWN profile == the EXACT dials (precise anchor). */
    out->cooldown.max_layer         = -1;
    out->cooldown.precision_tier    = 0;
    out->cooldown.mome_top_n        = -1;
    out->cooldown.skip_thresh_ratio = 0.0f;
    out->cooldown.kv_bits           = 0;
    out->cooldown.backend_hint      = 0;
    return 1;
}

void inferbit_burst_attach(inferbit_model* model, const ib_burst_config* cfg) {
    if (!model) return;
    ib_burst_ctrl* b = &model->burst;
    memset(b, 0, sizeof(*b));
    if (cfg) {
        /* Explicit API config wins verbatim (the public inferbit_burst_attach
         * contract). */
        b->cfg = *cfg;
        b->cfg.enabled = cfg->enabled ? 1 : 0;
    } else {
        /* NULL cfg (the load-time default call) → consult IB_BURST* env. With
         * IB_BURST unset this leaves the feature disabled (zero-init), so the
         * default run is byte-identical to today. */
        ib_burst_config env_cfg;
        if (ib_burst_env_config(&env_cfg)) {
            b->cfg = env_cfg;
        } else {
            b->cfg.enabled = 0;
        }
    }
    b->cur            = IB_PROFILE_EXACT;
    b->since_cooldown = 0;
    b->ema_accept     = 1.0f;   /* optimistic prior: assume drafts accepted */
    b->last_margin    = 0.0f;
    b->last_norm      = 0.0f;
    b->burst_steps    = 0;
    b->cooldown_steps = 0;
    b->burst_bytes    = 0;
    b->cooldown_bytes = 0;

    /* Start in the EXACT profile. The active skip ratio is seeded from the
     * EXACT profile's skip_thresh_ratio (0), then overridden by IB_PQV2_SKIP
     * so the default run matches today exactly. */
    model->active_profile           = &g_profile_exact;
    model->active_skip_thresh_ratio = g_profile_exact.skip_thresh_ratio;
    {
        float env_ratio = ib_burst_env_skip_ratio();
        if (env_ratio > 0.0f) model->active_skip_thresh_ratio = env_ratio;
    }
}

void inferbit_set_compute_profile(inferbit_model* model, ib_profile_kind kind) {
    if (!model) return;
    const ib_compute_profile* p;
    switch (kind) {
        case IB_PROFILE_BURST:    p = &model->burst.cfg.burst;    break;
        case IB_PROFILE_COOLDOWN: p = &model->burst.cfg.cooldown; break;
        case IB_PROFILE_EXACT:
        default:                  p = &g_profile_exact;           break;
    }
    model->active_profile           = p;
    model->active_skip_thresh_ratio = p->skip_thresh_ratio;
    model->burst.cur                = kind;
}

ib_profile_kind ib_burst_step_decide(inferbit_model* m) {
    if (!m) return IB_PROFILE_EXACT;
    ib_burst_ctrl* b = &m->burst;
    if (!b->cfg.enabled) {
        /* Feature off: never leave the exact path. */
        inferbit_set_compute_profile(m, IB_PROFILE_EXACT);
        return IB_PROFILE_EXACT;
    }

    /* Cool down when we've run a full burst window, or when either the
     * accept-rate EMA or the last logit margin has fallen below floor. */
    int cool = (b->cfg.cooldown_period > 0 &&
                b->since_cooldown >= b->cfg.cooldown_period) ||
               (b->cfg.accept_floor > 0.0f &&
                b->ema_accept < b->cfg.accept_floor) ||
               (b->cfg.margin_floor > 0.0f &&
                b->last_margin < b->cfg.margin_floor);

    ib_profile_kind kind;
    if (cool) {
        b->since_cooldown = 0;
        b->cooldown_steps++;
        kind = IB_PROFILE_COOLDOWN;
    } else {
        b->since_cooldown++;
        b->burst_steps++;
        kind = IB_PROFILE_BURST;
    }
    inferbit_set_compute_profile(m, kind);
    return kind;
}

void ib_burst_feed_accept(inferbit_model* m, int accepted, int drafted) {
    if (!m) return;
    ib_burst_ctrl* b = &m->burst;
    float sample = (drafted > 0) ? ((float)accepted / (float)drafted)
                                 : b->ema_accept;
    b->ema_accept = 0.9f * b->ema_accept + 0.1f * sample;
}

const ib_compute_profile* ib_active_profile(inferbit_model* m) {
    if (!m || !m->active_profile) return &g_profile_exact;
    return m->active_profile;
}

int ib_active_use_l2(inferbit_model* m) {
    const ib_compute_profile* p = ib_active_profile(m);
    /* L1+L2 (exact) unless the profile selects the L1-only coarse tier. */
    return (p->precision_tier == 1) ? 0 : 1;
}

int ib_active_max_layer(inferbit_model* m) {
    const ib_compute_profile* p = ib_active_profile(m);
    /* Full depth unless the profile requests early exit. */
    return p->max_layer;
}

void ib_burst_log_summary(inferbit_model* model) {
    if (!model) return;
    const char *e = getenv("IB_BURST_LOG");
    if (!(e && e[0] && e[0] != '0')) return;
    const ib_burst_ctrl* b = &model->burst;
    fprintf(stderr,
            "[ib-burst] enabled=%d period=%d burst_steps=%llu "
            "cooldown_steps=%llu accept_ema=%.3f last_margin=%.4f last_norm=%.4f\n",
            b->cfg.enabled, b->cfg.cooldown_period,
            (unsigned long long)b->burst_steps,
            (unsigned long long)b->cooldown_steps,
            b->ema_accept, b->last_margin, b->last_norm);
}

void inferbit_get_step_metrics(inferbit_model* model, float* out_margin,
                               float* out_hidden_norm, float* out_accept_rate) {
    if (!model) {
        if (out_margin)      *out_margin      = 0.0f;
        if (out_hidden_norm) *out_hidden_norm = 0.0f;
        if (out_accept_rate) *out_accept_rate = 0.0f;
        return;
    }
    const ib_burst_ctrl* b = &model->burst;
    if (out_margin)      *out_margin      = b->last_margin;
    if (out_hidden_norm) *out_hidden_norm = b->last_norm;
    if (out_accept_rate) *out_accept_rate = b->ema_accept;
}
