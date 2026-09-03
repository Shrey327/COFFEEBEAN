#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
rnnoise_commit=904a876dce1f9ab8860c0a5000ed151f9f6eef58
model_version=0b50c45
model_sha256=4ac81c5c0884ec4bd5907026aaae16209b7b76cd9d7f71af582094a2f98f4b43
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/coffeebean-rnnoise.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT HUP INT TERM

source_dir="$work_dir/source"
source_archive="$work_dir/source.tar.gz"
model_archive="$work_dir/rnnoise_data-$model_version.tar.gz"
output_dir="$project_dir/build/rnnoise-v0.2"
mkdir -p "$source_dir" "$output_dir"

curl -fL --retry 3 \
  "https://github.com/xiph/rnnoise/archive/$rnnoise_commit.tar.gz" \
  -o "$source_archive"
curl -fL --retry 3 \
  "https://media.xiph.org/rnnoise/models/rnnoise_data-$model_version.tar.gz" \
  -o "$model_archive"

if command -v shasum >/dev/null 2>&1; then
  actual_model_sha256=$(shasum -a 256 "$model_archive" | awk '{print $1}')
else
  actual_model_sha256=$(sha256sum "$model_archive" | awk '{print $1}')
fi
if [ "$actual_model_sha256" != "$model_sha256" ]; then
  echo "RNNoise model checksum mismatch" >&2
  exit 1
fi

tar -xzf "$source_archive" --strip-components=1 -C "$source_dir"
tar -xzf "$model_archive" -C "$source_dir"

# RNNoise's vector code expects this small Opus compatibility macro.
cat > "$source_dir/src/os_support.h" <<'EOF'
#ifndef RNNOISE_OS_SUPPORT_H
#define RNNOISE_OS_SUPPORT_H
#include <string.h>
#define OPUS_CLEAR(dst, n) memset((dst), 0, (n) * sizeof(*(dst)))
#endif
EOF

case $(uname -s) in
  Darwin)
    library="$output_dir/librnnoise.dylib"
    link_flag=-dynamiclib
    ;;
  Linux)
    library="$output_dir/librnnoise.so"
    link_flag=-shared
    ;;
  *)
    echo "Only macOS and Linux are supported" >&2
    exit 1
    ;;
esac

cc=${CC:-cc}
"$cc" -O3 -DDISABLE_DEBUG_FLOAT -DRNNOISE_BUILD -fPIC "$link_flag" \
  -I"$source_dir/include" -I"$source_dir/src" \
  "$source_dir/src/denoise.c" \
  "$source_dir/src/rnn.c" \
  "$source_dir/src/pitch.c" \
  "$source_dir/src/kiss_fft.c" \
  "$source_dir/src/celt_lpc.c" \
  "$source_dir/src/nnet.c" \
  "$source_dir/src/nnet_default.c" \
  "$source_dir/src/parse_lpcnet_weights.c" \
  "$source_dir/src/rnnoise_data_little.c" \
  "$source_dir/src/rnnoise_tables.c" \
  -lm -o "$library"

echo "Built RNNoise v0.2 little: $library ($(wc -c < "$library" | tr -d ' ') bytes)"
