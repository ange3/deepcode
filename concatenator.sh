tail -n +1 [0-9][0-9]*.txt > concatenated.csv
sed -i "1s/^==> //" concatenated.csv
((sed ':a;N;$!ba;s/\n/,/g' <concatenated.csv) | sed "s/,==> /\n/g") | sed "s/.txt <==//g" > concatenated.csv