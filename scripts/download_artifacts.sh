#!/usr/bin/env bash
# Downloads doctrine PDFs and demo-input source reports for the doctrinal AAR generator.
# Outputs:
#   corpus/raw/*.pdf   — doctrine corpus (JP, FM, ATP, AAR Leader's Guide)
#   demo_inputs/raw/   — wargame report PDFs (CSIS, RAND, CNAS) used as transcript sources
#
# Failure mode: if a URL 404s, skip with a warning. Do not block the whole download.
# Re-run is safe — curl --output skips overwrite-on-error and -f means no partial file on 404.

set -u
cd "$(dirname "$0")/.." || exit 1

mkdir -p corpus/raw demo_inputs/raw

fetch() {
    # fetch <out_path> <url> <label>
    local out="$1"
    local url="$2"
    local label="$3"
    if [ -f "$out" ] && [ -s "$out" ]; then
        echo "[skip] $label already present at $out"
        return 0
    fi
    echo "[get ] $label"
    if curl -fL --retry 3 --retry-delay 2 -A "Mozilla/5.0 (compatible; AAR-bot/0.1)" -o "$out" "$url"; then
        echo "[ok  ] $label -> $out"
    else
        echo "[FAIL] $label  ($url)"
        rm -f "$out"
        return 1
    fi
}

# ===== Doctrine corpus =====

# JP 5-0 — Joint Planning. Canonical planning doctrine; "branches and sequels" vocab.
fetch corpus/raw/jp5_0.pdf \
  "https://irp.fas.org/doddir/dod/jp5_0.pdf" \
  "JP 5-0 Joint Planning"

# JP 3-0 — Joint Operations.
fetch corpus/raw/jp3_0.pdf \
  "https://www.safety.marines.mil/Portals/92/Ground%20Safety%20for%20Marines%20(GSM)/References%20Tab/JP%203-0%20Joint%20Operations%20PDF.pdf" \
  "JP 3-0 Joint Operations"

# FM 3-0 — Operations.
fetch corpus/raw/fm3_0.pdf \
  "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN36290-FM_3-0-000-WEB-2.pdf" \
  "FM 3-0 Operations"

# FM 5-0 — Planning and Orders Production.
fetch corpus/raw/fm5_0.pdf \
  "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN35404-FM_5-0-000-WEB-1.pdf" \
  "FM 5-0 Planning and Orders Production"

# 2013 AAR Leader's Guide. CRITICAL — this is the format the generator targets.
fetch corpus/raw/aar_leaders_guide_2013.pdf \
  "https://pinnacle-leaders.com/wp-content/uploads/2018/02/Leaders_Guide_to_AAR.pdf" \
  "2013 AAR Leader's Guide"

# ATP 7-100.3 — Chinese Tactics. Adversary doctrine for red-cell sections.
fetch corpus/raw/atp7_100_3.pdf \
  "https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN34236-ATP_7-100.3-000-WEB-1.pdf" \
  "ATP 7-100.3 Chinese Tactics"

# ===== Demo-input sources =====

# CSIS — The First Battle of the Next War. Canonical demo input.
fetch demo_inputs/raw/csis_first_battle.pdf \
  "https://csis-website-prod.s3.amazonaws.com/s3fs-public/publication/230109_Cancian_FirstBattle_NextWar.pdf" \
  "CSIS First Battle of the Next War"

# RAND — Relighting Vulcan's Forge (Ryseff). Demo input #2; cited in README anti-patterns.
fetch demo_inputs/raw/vulcans_forge.pdf \
  "https://www.rand.org/content/dam/rand/pubs/research_reports/RRA2900/RRA2930-1/RAND_RRA2930-1.pdf" \
  "RAND Relighting Vulcan's Forge"

# CNAS — Dangerous Straits. Optional third demo input.
fetch demo_inputs/raw/cnas_dangerous_straits.pdf \
  "https://s3.amazonaws.com/files.cnas.org/CNAS+Report-Dangerous+Straits-Defense-Jun+2022-FINAL-print.pdf" \
  "CNAS Dangerous Straits"

echo ""
echo "===== Download summary ====="
echo "Corpus:"
ls -lh corpus/raw/ 2>/dev/null | tail -n +2
echo ""
echo "Demo inputs:"
ls -lh demo_inputs/raw/ 2>/dev/null | tail -n +2
