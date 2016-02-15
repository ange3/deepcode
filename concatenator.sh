tail -n +1 [0-9][0-9]*.txt > concatenated.txt
sed -i "1s/^==> //" concatenated.txt
((sed ':a;N;$!ba;s/\n/,/g' <concatenated.txt) | sed "s/,==> /\n/g") | sed "s/.txt <==//g" > concatenated_test.csv