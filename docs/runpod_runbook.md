# RunPod runbook

Setup: volume `chess-ai` (EU-RO-1, 30 GB) mounted at `/workspace`; on-demand L4 pod; code in
`/workspace/Chess_Bot`, artifacts in `/workspace/chess-ai`. Rules live in plan v4 Sec 1.

Everything under `/workspace` survives a pod stop. Everything else (pip packages, apt
installs, env vars, tmux sessions) is erased.

Constants (change here if they change): L4 rate **$0.59/hr**, balance floor **$2**
(enough for storage).

## A. Run something on the pod

Two independent parts. Part 1 is identical every time. Part 2 is the only part that is
specific to the job.

How they relate: the watchdog (Part 1) is just a timer on the pod. It does not know or care
what you run, and it does not read any notebook or script. When the timer ends it stops the
pod, which kills whatever is running. The job (Part 2) is a separate command. If the job
finishes early, nothing happens and you keep the pod. If you want to run several jobs, one
watchdog covers all of them.

### Part 1. Budget watchdog (same every time)

Start it once per session, right after the pod is up (before any long job), in the pod
terminal. Only `BAL` changes: your balance from the Billing page right now.

```
pkill -f "runpodctl stop pod"        # cancel any older watchdog (harmless if none)
BAL=12; FLOOR=2; RATE=0.59
SECS=$(python -c "print(int(($BAL-$FLOOR)/$RATE*0.9*3600))")
echo "pod will be stopped in $SECS seconds (~$(python -c "print(round($SECS/3600,1))") h)"
mkdir -p /workspace/chess-ai/logs
nohup bash -c "sleep $SECS; runpodctl stop pod $RUNPOD_POD_ID" > /workspace/chess-ai/logs/watchdog.log 2>&1 &
```

The 0.9 is a 10% safety margin. Example: `BAL=12` gives about 15.2 hours. If you top up the
balance or want more time, run the block again with the new `BAL`: it cancels the old
timer first. Running a stopped-and-restarted pod needs a new watchdog (section B).

### Part 2. The job (what is specific to each run)

What to decide per job:

| Job | Notebook | Set in the notebook (in the repo, then push) |
|---|---|---|
| Eval SFT checkpoints | `notebooks/02_train_sft.ipynb` | `RUN_TRAINING = False`, `EVAL_STEPS`, `OPPONENTS` |
| LLM self-play RL | `notebooks/04_train_llm_rl.ipynb` | `SMOKE_TEST` (True = 2 tiny rounds, no eval), `RUN_TRAINING`, `EVAL_ROUNDS` |
| AlphaZero (Step 9) | `notebooks/05_train_alphazero.ipynb` | `SMOKE_TEST` (True = 2 tiny rounds + timing probe, no eval), `RUN_TRAINING`, `EVAL_ROUNDS`, `OPPONENTS`; settings and the GPU-hour cap in `configs/alphazero.yaml` (section F) |
| Other (build dataset, etc.) | its own notebook | its own parameters cell |

Notebook parameters live in the notebook, and code is never edited on the pod, so change
them on your laptop, commit, push, then `git pull` on the pod.

Run it. Interactive (short jobs, you are watching): open the notebook in JupyterLab and
Run All. Headless (long jobs, you will walk away): in tmux, change only `NOTEBOOK`:

```
tmux new -s job
cd /workspace/Chess_Bot
NOTEBOOK=notebooks/04_train_llm_rl.ipynb
jupyter nbconvert --to notebook --execute $NOTEBOOK \
  --output-dir /workspace/chess-ai/logs \
  --output "$(basename $NOTEBOOK .ipynb)_$(date +%m%d_%H%M).ipynb" \
  --ExecutePreprocessor.timeout=-1
```

- Detach: `Ctrl-b`, release both keys, then `d`. Type nothing else in the session (no
  `Ctrl-d`, `Ctrl-c`, or `exit`: they can end the job). Close the browser freely.
  Reattach: `tmux attach -t job`.
- Check it is protected before walking away: `tmux ls` must list `job`. If it says "no
  sessions" while the job is still running, the job is unprotected (a closed terminal can
  kill it): stop it (`kill <nbconvert pid> <its kernel pid>`, not `pkill`) and relaunch
  in a fresh tmux session.
- Check the watchdog and timing: `ps -eo etimes,args | grep "sleep .*runpodctl stop pod" | grep -v grep | awk '{s=$5; sub(";","",s); printf "elapsed %.1f h, remaining %.1f h\n", $1/3600, (s-$1)/3600}'`.
- Live view: a headless run prints nothing until it ends. Watch the files instead, e.g.
  `watch -n 30 'column -s, -t < <run folder>/rl_log.csv | tail -15'`. The GPU check is
  `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv`.
- The executed copy (all printed output) goes to `/workspace/chess-ai/logs/`. Open it if a
  run fails.
- Check results any time: `cat /workspace/chess-ai/results/results.csv`.
- Training runs also write their own logs inside the run folder (for RL, `rl_log.csv`).

### What happens at the end

- Job finishes before the cap: nothing. The pod keeps running (and billing at $0.59/hr)
  until you stop it (section C) or the watchdog fires, whichever is first. If you are done,
  stop it yourself.
- Cap reached: the watchdog stops the pod. A job still running is killed. You lose at most
  the batch or round in progress (evals save after each batch; training saves every N).
- Anything written to `/workspace` is kept either way.

Download `results.csv` (and any new checkpoint you can't lose) to your laptop while the
pod is still up: JupyterLab file browser, right-click, Download. You can't browse the volume
once the pod is stopped.

## B. Restart after a pod has stopped

1. **Check the account.** Billing page: balance above your next run's cost plus the floor.
   Storage page: the `chess-ai` volume still exists (EU-RO-1, 30 GB).
2. **Deploy a new pod from the volume.** Pods, Deploy, choose L4 and on-demand. The
   volume must be attached at `/workspace` (the Storage section shows `chess-ai`,
   EU-RO-1). L4 availability is low, so retry later if it says no capacity.
3. **Set `HF_TOKEN`** as an environment variable on the pod (optional: the model is public).
4. **Open the JupyterLab terminal** and rebuild the pod-local setup:

   ```
   cd /workspace/Chess_Bot
   git fetch && git checkout <branch> && git pull
   apt-get update && apt-get install -y tmux
   pip install -r requirements.txt
   ```

5. **Verify:**

   ```
   nvidia-smi                       # should say L4
   ls /workspace/chess-ai/checkpoints/sft
   cat /workspace/chess-ai/results/results.csv
   ```

   Optionally run `notebooks/00_setup.ipynb` top to bottom (checks the GPU, Stockfish, and
   that the model loads).
6. **Start the watchdog** (section A, Part 1), then the job (Part 2).
7. The old stopped pod can be terminated from the Pods page once the new one is up. The
   volume is separate, so no data is lost, and a stopped pod may still bill for its disk.

## C. Stop a pod yourself

- From its terminal: `runpodctl stop pod $RUNPOD_POD_ID`.
- Or from the Pods page: Stop.
- Back up first: download `results.csv` to your laptop.

## D. Low-balance safety

- Turn on the low-balance email alert (no auto-pay).
- Keep the balance above the floor so the volume is never in arrears.
- After any long idle spell, open the Storage page and check the volume is still there.

## E. Planning a long run, and what can go wrong

### Does the watchdog fit the job?

- The GPU-hour counter is the same unit as wall-clock hours (one GPU). Wall-clock is the
  counted hours plus overhead: model load, checkpoint saves, the gap between rounds (roughly
  10 to 30 minutes over a 10-hour run).
- The cap is checked at the end of a round, so training can overshoot by up to one round.
- Budget for: training (cap + one round + overhead) + evals (unknown; plan 1 to 2 h or
  more) + the time between starting the watchdog and starting the job.
- The watchdog counts from the moment it was started, not from the job start. Start it
  right before the job.
- If the job ends early, the pod keeps running (and billing) until the watchdog fires or you
  stop it. It can never take the balance below roughly the floor plus the 10% margin.

### Outcomes and recovery (RL notebook; other notebooks follow the same pattern)

What is saved: RL saves the latest round with its resume state after every round, plus a
kept checkpoint every `checkpoint_every` rounds. Evals save `results.csv` after every batch
of games, and a checkpoint that is already evaluated is skipped. A resume with different
settings is refused (`run_config.json`), so do not edit the config or notebook parameters
between sessions.

| What happens | State on the volume | Recovery |
|---|---|---|
| **A.** Training hits the cap, all evals finish before the watchdog | Everything done; pod idles until the watchdog stops it | Nothing. Download `results.csv`. |
| **B.** Training done, watchdog fires mid-eval (the likeliest problem) | Training complete; finished evals saved; the one in progress loses at most one batch (under 10 games) | Top up. Section B (restart), then start a watchdog and rerun the **same notebook**. Training sees it is finished, and evals continue from where they stopped. To skip the training check entirely, set `RUN_TRAINING = False`. |
| **C.** Watchdog fires mid-training (rounds far slower than planned) | Latest completed round and resume state saved; the round in progress is lost | Same restart. Training resumes from the latest round, counts the GPU-hours already spent, and the cap still applies. |
| **D.** Job crashes (out of memory, error) | Same as C | Open the executed notebook in `logs/` (or the tmux output) for the error, fix on the laptop, push, pull, rerun. |
| **E.** Pod lost | Volume intact | Section B. |

Rule of thumb: rerunning the same notebook with the same settings is always the recovery.
Nothing already finished is redone.

## F. AlphaZero (Step 9): smoke run, then stage 1

### One copy of the code per volume

`/workspace/Chess_Bot` is on the network volume, so every pod attached to it shares that one
folder. Checking out a branch there changes the code under any job another pod is running
(modules that load late, like an eval cell's imports, would pick up the new branch). Run
AlphaZero only when no other pod is running a job from it. To train AlphaZero alongside an
LLM job later, first decide with Claude whether AlphaZero gets its own clone (e.g.
`/workspace/Chess_Bot_az`); that needs a one-line change to notebook 05's `REPO_DIR`.

### Smoke run (~10-20 minutes, well under $1)

1. **Pod.** Reuse a pod that is up and idle (`tmux ls` lists no running job, `nvidia-smi`
   shows no process), or start one (section B). It must be an L4.
2. **Watchdog** (section A, Part 1) with your current balance.
3. **Code** (nothing else may be running from this folder):

   ```
   cd /workspace/Chess_Bot
   git fetch && git checkout step-9-alphazero && git reset --hard origin/step-9-alphazero
   git log -1 --oneline
   pip install -q -r requirements.txt
   python -m pytest tests/ -q          # ~1 min; Stockfish tests run here, the Qwen download test skips
   ```

4. **Run notebook 05 headless** with `SMOKE_TEST = True` (the smoke run was done Oct 31 and
   the committed notebook now says `False`; set it back on the laptop, commit, push, pull to
   smoke-test again, e.g. after a code change):

   ```
   tmux new -s job
   cd /workspace/Chess_Bot
   NOTEBOOK=notebooks/05_train_alphazero.ipynb
   jupyter nbconvert --to notebook --execute $NOTEBOOK \
     --output-dir /workspace/chess-ai/logs \
     --output "$(basename $NOTEBOOK .ipynb)_$(date +%m%d_%H%M).ipynb" \
     --ExecutePreprocessor.timeout=-1
   ```

   It plays 2 rounds of 8 games (8 simulations, 40-ply cap) into
   `checkpoints/alphazero/az_b6c64_s0_smoke`, then times one move of 128 games at 100
   simulations and one update step at batch 256, with a network of the real size. No eval,
   no plot, nothing written to `results.csv`.

5. **Send Claude the printed output.** This prints every cell's text output from the newest
   executed copy:

   ```
   python - <<'PY'
   import glob, json
   path = sorted(glob.glob("/workspace/chess-ai/logs/05_train_alphazero_*.ipynb"))[-1]
   print(path)
   for c in json.load(open(path))["cells"]:
       for o in c.get("outputs", []):
           text = "".join(o.get("text", "")) or "".join(o.get("data", {}).get("text/plain", ""))
           print(text)
   PY
   ```

   The numbers that matter: GPU / CPU / cores; seconds per round (self-play vs. update); the
   probe's seconds per move of 128 games and milliseconds per update step; the projected
   rounds in 9 GPU-h; peak GPU memory. An error anywhere is also worth sending as is.

6. **Stop the pod** if nothing else is planned (section C).

### After the smoke run: settings, then stage 1

Settings confirmed Oct 31 (plan Sec 5 item 4): the provisional defaults with 50 simulations
per move, and notebook 05 committed with `SMOKE_TEST = False`. Stage 1 on the pod: steps 1-3
above (with the stage-1 `BAL` below), then step 4's command; `az_b6c64_s0_smoke` is separate,
so the real run (`az_b6c64_s0`) starts fresh. Watch it with
`watch -n 60 'column -s, -t < /workspace/chess-ai/checkpoints/alphazero/az_b6c64_s0/az_log.csv | tail -8'`.

Budget for stage 1 (`max_gpu_hours: 9`): about 9 h of training + up to one round past the
cap + evals (up to 4 checkpoints x 60 games; plan 1-3 h, as search makes AlphaZero slower
per move than the LLM) + setup. At $0.59/h that is roughly 12-13 h, so `BAL` of about $10 or
more (the watchdog keeps a 10% margin above the $2 floor). If the watchdog fires before the
evals finish, rerun the same notebook (section E, outcome B). Raising the cap to 18 (stage
2) is a config change + push + rerun: the run resumes.

