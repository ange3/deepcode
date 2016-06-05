ls *.json > filelist.txt

tail -n +1 [0-9]*.json > coalesced.json

sed "s/^==> 0.json <==/[/" coalesced.json > array_start.json

(sed "s/==> [0-9][0-9]*.json <==/,/" array_start.json) > concatenated_ASTs.json

sed -ie "\$a]" concatenated_ASTs.json

#sed "s/==> [0-9]*.json <==\n{\n/,{\n\t\"astID\": \"&\",/" concatenated_ASTs.json