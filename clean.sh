sed 's/#.*//' requirements.txt | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | grep -v '^$' > clean_reqs.txt
