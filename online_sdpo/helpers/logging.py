def print_first_example(chat, feedback):
    print("=" * 60, flush=True)
    print("[DEBUG] User simulator — first example", flush=True)
    print(f"  FULL FORMATTED INPUT:\n{chat}", flush=True)
    print(f"  OUTPUT feedback:\n{feedback}", flush=True)
    print("=" * 60, flush=True)

def print_train_step(metrics) -> None:
    step = metrics["step"]
    loss = metrics.get("loss", 0.0)
    logp = metrics.get("policy_logp_mean", 0.0)
    train_t = metrics.get("train_time_s", 0.0)
    print(
        f"[STEP {step:4d}] batch={metrics['batch_size']} "
        f"loss={loss:.6f}  policy_logp={logp:.4f}  "
        f"gen={metrics['gen_time_s']:.1f}s  sim={metrics['sim_time_s']:.1f}s  "
        f"train={train_t:.1f}s",
        flush=True,
    )

def print_first_judge(chat, d, judge):
    print("=" * 60, flush=True)
    print("[DEBUG] Judge — first eval, first example", flush=True)
    print(f"  FULL FORMATTED INPUT:\n{chat}", flush=True)
    print(f"  OUTPUT decision[0]: {d[0] if d else 'N/A'}", flush=True)
    raw = getattr(judge, "last_raw_outputs", None)
    if raw:
        print(f"  JUDGE REASONING[0]:\n{raw[0][:1000]}", flush=True)
    print("=" * 60, flush=True)

def print_eval_result(metrics, gen_time, judge_time):
    wr = metrics.get("winrate_a", float("nan"))
    se = metrics.get("se", float("nan"))
    wr_inc_ties = (metrics['wins_a'] + 0.5 * metrics['ties']) / max(metrics['n'], 1) * 100.0
    print(
        f"[EVAL @ step {metrics['step']}] winrate={wr:.1f}% ±{se:.1f}  "
        f"wins={metrics['wins_a']} losses={metrics['wins_b']} "
        f"ties={metrics['ties']}  "
        f"wr_inc_ties={wr_inc_ties:.1f}%  "
        f"gen={gen_time:.1f}s  judge={judge_time:.1f}s",
        flush=True,
    )
