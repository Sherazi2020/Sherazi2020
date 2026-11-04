#!/usr/bin/env bash
# Usage: ./render.sh [seconds]   -> renders deep_calm_sample.mp4 (1080p, H.264 CRF 18, AAC 256k)
set -e; D=${1:-60}
python3 make_music.py "$D"
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
ffmpeg -y -hide_banner -loglevel error \
 -f lavfi -i "gradients=s=480x270:r=30:d=$D:c0=0x0b1a3a:c1=0x1f6f78:c2=0x5b3c88:c3=0x0e2a47:c4=0x2a9d8f:n=5:speed=0.004:type=radial" \
 -i music.wav \
 -filter_complex "[0:v]gblur=sigma=25,scale=1920:1080:flags=bicubic,noise=alls=4:allf=t,vignette=PI/4,format=yuv420p,drawtext=fontfile=$FONT:text='Deep Calm':fontcolor=white@0.85:fontsize=72:x=(w-tw)/2:y=(h-th)/2-30:alpha='if(lt(t,2),0,if(lt(t,5),(t-2)/3,if(lt(t,10),1,if(lt(t,13),(13-t)/3,0))))',drawtext=fontfile=$FONT:text='music for rest and recovery':fontcolor=white@0.6:fontsize=34:x=(w-tw)/2:y=(h/2)+50:alpha='if(lt(t,3),0,if(lt(t,6),(t-3)/3,if(lt(t,10),1,if(lt(t,13),(13-t)/3,0))))',fade=in:0:60,fade=out:st=$((D-3)):d=3[v]" \
 -map "[v]" -map 1:a -c:v libx264 -preset slow -crf 18 -c:a aac -b:a 256k -shortest -movflags +faststart deep_calm_sample.mp4
