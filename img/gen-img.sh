#!/bin/bash
set -auxe
CMD="node ../web/shot.mjs --size 800x600"
PREF0="#src=models/penguin.3mf"
PREF="${PREF0}&cam=197,116,75,175,157,44&cut=,47"
${CMD} "${PREF}" --panel src source.png

PREF="${PREF}&run=1"
${CMD} "${PREF}" normal.png
${CMD} "${PREF}&core=1" core.png
${CMD} "${PREF}&core=1&primer=4" core_primer.png
${CMD} "${PREF}&shell=5" shell.png

PREF="${PREF0}&cam=234.56,-4.42,69.12,172.33,154.96,13.66&run=1&bg=003333"
${CMD} "${PREF}&inlay=1,2,4" inlays-keep.png
${CMD} "${PREF}&inlay=1,2,4&ileft=body" inlays-body.png
