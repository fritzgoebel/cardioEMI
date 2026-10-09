#!/bin/bash
# sync_to_karolina.sh - back-compat wrapper; use sync_to_cluster.sh <id> for any cluster.
exec bash "$(cd "$(dirname "$0")" && pwd)/sync_to_cluster.sh" karolina
