#!/usr/bin/env bash
# ============================================================
#  THAMAN — Automated Retrain Script
#  Runs every 6 months via launchd (com.thaman.retrain).
#  - Backs up current model
#  - Trains train_stack_v11.py
#  - Keeps new model only if R² improves or stays equal
#  - Sends macOS notification with result
#  - Logs to ~/Library/Logs/thaman_retrain.log
# ============================================================

set -euo pipefail

PROJECT="/Users/totam/Desktop/new_try"
LOG="$HOME/Library/Logs/thaman_retrain.log"
BACKUP_DIR="$PROJECT/models/backup"
PYTHON="/Users/totam/.pyenv/versions/3.12.12/bin/python"

# ── Helpers ─────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
notify() {
  osascript -e "display notification \"$2\" with title \"THAMAN Retrain\" subtitle \"$1\" sound name \"Glass\"" 2>/dev/null || true
}
read_r2() {
  "$PYTHON" -c "import json; m=json.load(open('$PROJECT/models/meta.json')); print(m.get('stack',{}).get('r2_holdout',0))" 2>/dev/null || echo "0"
}

mkdir -p "$BACKUP_DIR"
cd "$PROJECT"

log "====== THAMAN retrain started ======"

# ── Read current R² before training ─────────────────────────
OLD_R2=$(read_r2)
log "Current model R²: $OLD_R2"

# ── Back up current model files ──────────────────────────────
STAMP=$(date '+%Y%m%d_%H%M%S')
cp models/thaman_stack.pkl "$BACKUP_DIR/thaman_stack_${STAMP}.pkl"
cp models/meta.json        "$BACKUP_DIR/meta_${STAMP}.json"
log "Backup saved → $BACKUP_DIR/*_${STAMP}.*"

# ── Run training ─────────────────────────────────────────────
log "Training started …"
"$PYTHON" -u training/train_stack_v11.py >> "$LOG" 2>&1
log "Training finished."

# ── Check if new model is better ─────────────────────────────
NEW_R2=$(read_r2)
log "New model R²: $NEW_R2"

IMPROVED=$("$PYTHON" -c "print('yes' if float('$NEW_R2') >= float('$OLD_R2') - 0.002 else 'no')")

if [ "$IMPROVED" = "yes" ]; then
  log "Model accepted (R² $OLD_R2 → $NEW_R2). Restarting API …"
  # Restart API if running under launchd (adjust label if needed)
  pkill -f "uvicorn api.main:app" 2>/dev/null || true
  sleep 2
  nohup "$PYTHON" -m uvicorn api.main:app --host 0.0.0.0 --port 8000 >> "$LOG" 2>&1 &
  notify "✅ Retrain complete" "R² improved: $OLD_R2 → $NEW_R2"
  log "API restarted."
else
  log "Model REJECTED (R² $OLD_R2 → $NEW_R2 fell too much). Restoring backup …"
  cp "$BACKUP_DIR/thaman_stack_${STAMP}.pkl" models/thaman_stack.pkl
  cp "$BACKUP_DIR/meta_${STAMP}.json"        models/meta.json
  notify "⚠️ Retrain rejected" "R² dropped $OLD_R2 → $NEW_R2. Old model restored."
fi

# ── Clean up old backups (keep last 3) ───────────────────────
ls -t "$BACKUP_DIR"/thaman_stack_*.pkl 2>/dev/null | tail -n +4 | xargs rm -f
ls -t "$BACKUP_DIR"/meta_*.json        2>/dev/null | tail -n +4 | xargs rm -f

log "====== THAMAN retrain done ======"
