#!/usr/bin/env bash
# The zstd C library in the default builds (CLB build): exactly the GeoTIFF/COG path, nothing else.
#
# - Natively, the default graphs of terrio, terrio-cli, terrio-py and terrio-js may hold the zstd
#   crates only as terrio's `libzstd-geotiff` brings them: `zstd` from terrio itself and from the
#   `tiff` crate's `zstd` feature, which only terrio enables, then `zstd-safe` and `zstd-sys`.
# - terrio with every default feature but `libzstd-geotiff` holds none (so no other feature, and no
#   other dependency, brings them).
# - The WebAssembly graphs (the npm package, Pyodide) hold none: `libzstd-geotiff` enables nothing
#   there.
set -euo pipefail
cd "$(dirname "$0")/.."

ZSTD='^(zstd|zstd-safe|zstd-sys)$'
fail() { echo "error: $*" >&2; exit 1; }

defaults=$(sed -n 's/^default = \[\(.*\)\]$/\1/p' crates/terrio/Cargo.toml | tr -d ' "')
case ",$defaults," in *,libzstd-geotiff,*) ;; *) fail "libzstd-geotiff is not a default feature of terrio";; esac
pure=$(echo ",$defaults," | sed 's/,libzstd-geotiff,/,/; s/^,//; s/,$//')

for p in terrio terrio-cli terrio-py terrio-js; do
  # Each zstd crate's dependents ("parent -> child"), natively and in WebAssembly together.
  edges=$(cargo tree --locked -p "$p" -e normal,build --target all --prefix depth | awk -v re="$ZSTD" '{
    match($0, /^[0-9]+/); d = substr($0, 1, RLENGTH) + 0; split(substr($0, RLENGTH + 1), a, " "); s[d] = a[1]
    if (a[1] ~ re) print s[d - 1] " -> " a[1] }' | sort -u)
  unexpected=$(echo "$edges" | grep -vE '^(terrio -> zstd|tiff -> zstd|zstd -> zstd-safe|zstd-safe -> zstd-sys)$' | grep . || true)
  [ -z "$unexpected" ] || fail "$p: zstd crates outside the GeoTIFF C path: $unexpected"
  # Who enables the `tiff` crate's Zstandard: terrio only.
  enablers=$(cargo tree --locked -p "$p" -e normal,build,features --target all -i tiff --prefix depth | awk '
    /^[0-9]+/ { match($0, /^[0-9]+/); d = substr($0, 1, RLENGTH) + 0; n = substr($0, RLENGTH + 1)
      if (n ~ /^tiff feature "zstd"/) { z = d; next }
      if (z >= 0 && d == z + 1) { split(n, a, " "); print a[1] } else if (d <= z) z = -1 }
    BEGIN { z = -1 }' | sort -u)
  others=$(echo "$enablers" | grep -vx 'terrio' | grep . || true)
  [ -z "$others" ] || fail "$p: the tiff crate's zstd feature is enabled by: $others"
  for t in wasm32-unknown-unknown wasm32-unknown-emscripten; do
    found=$(cargo tree --locked -p "$p" -e normal,build --target "$t" --prefix none | cut -d' ' -f1 | sort -u | grep -E "$ZSTD" || true)
    [ -z "$found" ] || fail "$p ($t): zstd crates in WebAssembly: $found"
  done
  echo "$p: $(echo "$edges" | tr '\n' ';')"
done

found=$(cargo tree --locked -p terrio -e normal,build --target all --no-default-features --features "$pure" --prefix none | cut -d' ' -f1 | sort -u | grep -E "$ZSTD" || true)
[ -z "$found" ] || fail "terrio without libzstd-geotiff ($pure): $found"
echo "terrio without libzstd-geotiff: no zstd crate"
