#!/bin/bash
# Download the cross-league test matches (reviewer comment R1.5) from the
# official league channels on YouTube, with the same settings as the
# LaLiga corpus (H.264 video <= 1080p + best audio, merged to mp4).
# Videos at 50 fps are resampled to 25 fps like the rest of the corpus.
set -e
source /home/mrt/Projects/Soccer-Video-Summarization/venv/bin/activate
OUTDIR="/mnt/Data-fast/mrt/lf/SoccerNetClip10Videos/OtherLeagues_HR"

declare -A MATCHES=(
    ["PL1"]="X6VNTk-Vl5w"      # Premier League 2014/15: Tottenham 5-3 Chelsea
    ["SA1"]="ZPXbqoiM4PY"      # Serie A 2024/25: Inter 4-4 Juventus
    ["BL1"]="1yHWGw8DH4A"      # Bundesliga: RB Leipzig 4-5 Bayern Munich
    ["L1"]="ZEGyZkJe-60"       # Ligue 1 2015/16: Troyes 0-9 PSG
)

for MATCH in PL1 SA1 BL1 L1; do
    VID="${MATCHES[$MATCH]}"
    DIR="$OUTDIR/$MATCH"
    mkdir -p "$DIR"
    if ls "$DIR"/*.mp4 1>/dev/null 2>&1; then echo "[$MATCH] exists"; continue; fi
    echo "[$MATCH] downloading $VID at $(date)"
    yt-dlp -f "bestvideo[height<=1080][vcodec^=avc1]+bestaudio[ext=m4a]/best[height<=1080]" --merge-output-format mp4 \
        -o "$DIR/%(title)s [%(id)s].%(ext)s" "https://www.youtube.com/watch?v=${VID}"
    F=$(ls "$DIR"/*.mp4 | head -1)
    FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$F")
    if [ "$FPS" != "25/1" ]; then
        echo "[$MATCH] resampling $FPS -> 25 fps"
        ffmpeg -loglevel error -i "$F" -vf fps=25 -c:v libx264 -crf 18 -preset fast -c:a copy "${F%.mp4}.25fps.mp4"
        mkdir -p "$DIR/original" && mv "$F" "$DIR/original/"
    fi
done
echo "=== done $(date) ==="
