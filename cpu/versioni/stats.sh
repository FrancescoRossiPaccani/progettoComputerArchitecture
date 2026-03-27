#!/bin/bash

# ===============================
# Stats.sh corretto per resume CSV
# ===============================

OUTPUT_FILE="log_run_v1.txt"

# 📄 crea file con header solo se non esiste
if [ ! -f "$OUTPUT_FILE" ]; then
  echo "image,threads,kernel,bucket,run,time_ms,error" > "$OUTPUT_FILE"
fi

IMAGES=("img1" "img2" "img3")
THREADS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
KERNEL=5
BUCKET=256
RUNS=10

# 🔢 totale run teorici
TOTAL=$(( ${#IMAGES[@]} * ${#THREADS[@]} * RUNS ))
COUNT=0

for img in "${IMAGES[@]}"; do
  for t in "${THREADS[@]}"; do
    for ((i=1;i<=RUNS;i++)); do
      
      COUNT=$((COUNT+1))
      PERCENT=$(( 100 * COUNT / TOTAL ))

      echo "[${PERCENT}%] Running $img | threads=$t | run=$i ($COUNT/$TOTAL)"

      # 🔁 skip se già eseguito
      if grep -q "^$img,$t,$KERNEL,$BUCKET,$i," "$OUTPUT_FILE"; then
        echo "⏭️  Skipping già eseguito: $img | threads=$t | run=$i"
        continue
      fi

      # ▶️ esecuzione del programma
      result=$(./v1 "$img" "$t")

      # ⏱ parsing output
      time=$(echo "$result" | grep "Tempo" | awk '{print $2}')
      error=$(echo "$result" | grep "Errore" | awk '{print $2}')

      # 💾 salva solo se time ed error non sono vuoti
      if [ -n "$time" ] && [ -n "$error" ]; then
          FULL_LINE="$img,$t,$KERNEL,$BUCKET,$i,$time,$error"
          echo "$FULL_LINE" >> "$OUTPUT_FILE"
          sync  # forza scrittura su disco
      else
          echo "⚠️  Output incompleto, skip riga: $img | threads=$t | run=$i"
      fi

      # ⏸ opzionale: pausa tra run
      sleep 1
    done
  done
done

# 🔹 ordina CSV per thread e run numericamente
# (esclude header dalla riga 1)
(head -n1 "$OUTPUT_FILE" && tail -n +2 "$OUTPUT_FILE" | sort -t, -k2n -k5n) > tmpfile && mv tmpfile "$OUTPUT_FILE"
echo "scritto"

echo "✅ COMPLETATO 100% ($TOTAL run)"