awk '{print($1,$2,$3+(($2)*($2)/1000))}' exp/f1/time_te1_l.list|sort -nk3
