#!/usr/bin/env bash
filename='xiao-chi.txt'

exec < $filename

while read line 
do
    read -r -a array <<< $line
    if [ "$1" == "get-link" ]; then
        echo "Get links for ${array[0]} - ${array[1]}"
        echo "python3 link-parser.py ${array[0]} 450 ${array[1]} "
    elif [ "$1" == "get-image" ]; then
        echo "Download images for ${array[0]} - ${array[1]}"
        bash curl.sh ${array[1]}
    else 
        echo "Get image or link"
    fi
done
