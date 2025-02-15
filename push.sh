#!/bin/bash

commit_message='newest'

if [ -z "$commit_message" ]; then
  commit_message="update the newest"
fi

git config --global credential.helper store

git init
git add .
git commit -m "$commit_message"

git remote add origin https://github.com/weeadd/videowebarena.git
git push -u origin main -f