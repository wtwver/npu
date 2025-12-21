#!/bin/sh
set -eu

cd "$(dirname "$0")"

usage() {
  cat <<EOF
usage: ./run.sh [--gdb] [op] [op-args...]

ops:
  recip abs floor ceil

examples:
  ./run.sh recip
  ./run.sh abs 4
  ./run.sh --gdb floor 1
EOF
}

GDB=0
case "${1:-}" in
  --gdb) GDB=1; shift ;;
esac
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

OP="${1:-recip}"
if [ $# -gt 0 ]; then shift; fi

CPP="${OP}.cpp"
BIN="./${OP}"

if [ ! -f "${CPP}" ]; then
  echo "unknown op '${OP}' (missing ${CPP})" >&2
  usage
  exit 2
fi

# if [ -f generate.py ]; then
#   GEN_SIZE=1
#   case "${1:-}" in
#     ''|*[!0-9]*) ;;
#     *) GEN_SIZE="$1" ;;
#   esac
#   python3 generate.py "${OP}" "${GEN_SIZE}" >/dev/null
# fi

g++ -o "${OP}" "${CPP}" -I../include -lrknnrt -std=c++11

: > run_output.txt
if [ "${GDB}" = "1" ]; then
  gdb -x rknn.gdb --args "${BIN}" "$@" | tee run_output.txt
else
  "${BIN}" "$@" | tee run_output.txt
fi
