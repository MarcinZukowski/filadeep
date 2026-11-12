#!/bin/bash
set -auxe
CMD="node ../web/shot.mjs --size 800x600"
PREF="#src=models/penguin.3mf&cam=197,116,75,175,157,44&cut=,47"

${CMD} "${PREF}" --panel src source.png

PREF="${PREF}&run=1"
${CMD} "${PREF}" normal.png
${CMD} "${PREF}&core=1" core.png
${CMD} "${PREF}&core=1&primer=4" core_primer.png
${CMD} "${PREF}&shell=5" shell.png
