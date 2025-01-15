#!/usr/bin/env bash

CYAN='\033[0;36m'
YELLOW='\033[0;33m'
NO_COLOR='\033[0m'

if [[ $# -ne 2 ]]; then
    echo 'Too many/few arguments, expecting two' >&2
    exit 1
fi

filename=$1
yt_url=$2

echo -e "${YELLOW}---processing---${NO_COLOR}"
echo $filename
echo $yt_url

yt-dlp -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4' -S vcodec:h264,res,acodec:m4a -o "${filename}" "${yt_url}" 
