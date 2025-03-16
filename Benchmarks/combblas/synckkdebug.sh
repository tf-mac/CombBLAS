#!/bin/bash
WORKFOLDER=tfcombblas-minor2
REMOTE_USER="exouser"
REMOTE_HOST="149.165.155.206"
REMOTE_PATH="/media/volume/workspace/kk/$WORKFOLDER"

# Define SSH key (optional if using default ~/.ssh/id_rsa)
SSH_KEY_PATH="~/.ssh/id_rsa_kl23395"
# Use rsync to transfer the folder
function syncfromdebug {
    echo "syncing from debug node"
    rsync -avz -e "ssh -i $SSH_KEY_PATH" \
    --exclude "debug-*" \
    --exclude "release-*" \
    --exclude ".git" \
    --exclude ".cache" \
    --delete \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" $WROOT/kk/
}

syncfromdebug