#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

docker run --rm -d -p 3000:3000 \
  --network development \
  -v $(pwd)/metabase-data:/metabase-data \
  -e "MB_DB_FILE=/metabase-data/metabase.db" \
  --name metabase metabase/metabase

exit 0
