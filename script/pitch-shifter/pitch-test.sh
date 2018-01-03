#!/bin/bash

pitchlevel=2

for i in ./test/**/mix0-*
do
      if [ -f $i ]
      then
          file=$(basename $i)
          newdir="compress/$(dirname $i)"
          echo $file
          #echo $newdir
          result=`python /sdc1/git/pitch-shifter-py/pitchshifter/pitchshifter.py -s $i -o $newdir/"pitch2-$file.wav" -p 2 -b 1`
      fi
done

