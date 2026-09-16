#!/bin/bash
git fetch --all --prune && git checkout -B "$(git rev-parse --abbrev-ref HEAD)" "origin/$(git rev-parse --abbrev-ref HEAD)"

