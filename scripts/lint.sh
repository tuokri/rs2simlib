#!/usr/bin/env bash
declare -i result=0

flake8 rs2simlib --count --select=E9,F63,F7,F82 --show-source --statistics
result+=$1

flake8 rs2simlib --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
result+=$1

if [ "$result" -gt 0 ]; then
  exit 1
else
  exit 0
fi
