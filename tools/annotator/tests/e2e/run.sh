#!/bin/bash
# Reinicia el servidor de pruebas (puerto 8792) con un data-root limpio y ejecuta el e2e.
SP=/tmp/annotator_e2e
DATA=$SP/annot_test_sbd
PIDS=$(ss -ltnp 2>/dev/null | awk '/:8792 /' | grep -o 'pid=[0-9]*' | cut -d= -f2)
[ -n "$PIDS" ] && kill $PIDS && sleep 0.5
mkdir -p $DATA/cache; find $DATA -mindepth 1 -maxdepth 1 ! -name cache -exec rm -rf {} +
cd /home/mrt/Projects/Soccer-Video-Summarization && source venv/bin/activate
nohup python tools/annotator/server.py --port 8792 --data-root $DATA > $SP/server_sbd.log 2>&1 &
for i in $(seq 30); do curl -sf localhost:8792/api/health >/dev/null && break; sleep 0.3; done
cd $SP/e2e_sbd && DATA_ROOT=$DATA node test_sbd.js
