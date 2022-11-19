search_dir=data/image
i=0
for entry in "$search_dir"/*
do
    if [ "$((i += 1))" -gt 1000 ] ; then
        break
    fi
    echo "$entry"
    python -m scripts.genvideo $entry --crop_fraction 5 --max_length 75
done
