#!/usr/bin/env bash
#Render paper/paper.md into a JOSS-styled paper/paper.pdf.
#
#Uses openjournals/inara, the same container JOSS runs to typeset accepted
#papers (it is what "@editorialbot generate pdf" invokes in a review issue),
#so the output matches what the journal publishes.
#
#The first run pulls a ~1-2 GB image and will be slow; later runs are fast.
#
#This user is not in the "docker" group, so the script escalates with sudo and
#you will be prompted for a password. To get rid of the prompt permanently:
#    sudo usermod -aG docker $USER
#then log out and back in (a new login session is required for the group to
#take effect) and re-run this script.

#To run call "! ./paper/build_pdf.sh"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

#Resolve the invoking user's ids BEFORE any escalation. Passing these to
#--user is what keeps paper.pdf owned by you; evaluated inside the sudo
#context they would resolve to 0:0 and the output would land owned by root.
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

DOCKER=(docker)
if ! docker info > /dev/null 2>&1; then
    echo "Docker needs elevated privileges for this user - sudo will ask for your password."
    DOCKER=(sudo docker)
fi

"${DOCKER[@]}" run --rm \
    --volume "$REPO_ROOT":/data \
    --user "$HOST_UID:$HOST_GID" \
    --env JOURNAL=joss \
    openjournals/inara:latest \
    -o pdf paper/paper.md

echo
echo "Wrote $REPO_ROOT/paper/paper.pdf"
