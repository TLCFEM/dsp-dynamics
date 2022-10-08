set term tikz size 14cm,5cm
set key opaque samplen 1 width -2 right bottom Left
set grid
set xrange [0:40]
set yrange [-1.2:1.7]
set xtics 0,4,40
set ytics -1,.5,1.5
set ylabel "roof drift (\\si{\\percent})"
set xlabel "time (\\si{\\second})"
set output "FRAME.tex"
plot "FRAME" u 1:($2/3.6) w lp pointinterval 101 lw 2 lc rgb "#e41a1c" title "fibre element", \
     "FRAME" u 1:($3/3.6) w lp pointinterval 107 lw 2 lc rgb "#377eb8" title "proposed element"
set output