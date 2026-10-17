# RunPod runbook

Setup: volume `chess-ai` (EU-RO-1, 30 GB) mounted at `/workspace`; on-demand L4 pod; code in
`/workspace/Chess_Bot`, artifacts in `/workspace/chess-ai`. Rules live in plan v4 Sec 1.

Everything under `/workspace` survives a pod stop. Everything else (pip packages, apt
installs, env vars, tmux sessions) is erased.

## A. Launch a run with a budget cap (watchdog stops the pod at the floor)

The cap is a timer that starts at launch. The job runs on its own. If the job finishes
early, nothing happens and you keep the pod (no migrating to a new L4). If the timer
reaches the cap, the pod is stopped, which also ends any job still running. Either way the
balance stays above the floor.

1. Pod terminal: `cd /workspace/Chess_Bot && git pull && cat /workspace/chess-ai/results/results.csv`
   (check the code is current and the results file is there).
2. Read your balance on the Billing page and the L4 $/hr on the pod. Decide the floor
   (default $2, enough for storage).
3. Start the watchdog, then the job in tmux (fill in `BAL`, `RATE`, and the notebook name):

   ```
   mkdir -p /workspace/chess-ai/logs
   tmux new -s eval
   cd /workspace/Chess_Bot
   BAL=12; FLOOR=2; RATE=0.39
   SECS=$(python -c "print(int(($BAL-$FLOOR)/$RATE*0.9*3600))")
   echo "pod will be stopped in $SECS seconds"
   nohup bash -c "sleep $SECS; runpodctl stop pod $RUNPOD_POD_ID" > /workspace/chess-ai/logs/watchdog.log 2>&1 &
   jupyter nbconvert --to notebook --execute notebooks/XX.ipynb \
     --output-dir /workspace/chess-ai/logs --ExecutePreprocessor.timeout=-1
   ```

4. Detach: `Ctrl-b`, then `d`. Close the browser freely.
5. Check progress any time: `cat /workspace/chess-ai/results/results.csv`.
   Reattach: `tmux attach -t eval`.

What happens:
- Job finishes before the cap: nothing. The pod keeps running (and billing) until you stop
  it (section C) or the watchdog fires at the cap time, whichever is first. If you are done,
  stop it yourself.
- Cap reached: the watchdog stops the pod. A job still running is killed; you lose at most
  the batch in progress (rows save after each batch; training saves every N steps).
- Anything written to `/workspace` is kept either way.

Changed your mind (topped up the balance, want more time)? Cancel the watchdog with
`pkill -f "runpodctl stop pod"`, then start a new one with the new `BAL`.

Download `results.csv` to your laptop while the pod is still up (JupyterLab file browser,
right-click, Download). You can't browse the volume once the pod is stopped.

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
6. Launch the next run with section A.
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
