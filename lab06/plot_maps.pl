#!/usr/bin/gnuplot
set term png

set output "map1.png"
set title "nx = ny = 50, eps1 = eps2 = 1.0"
set xlabel "x"
set ylabel "y"
set pm3d map
set palette defined (-10 "blue", 0 "white", 10 "red")
set size ratio -1
splot [0:5][0:5] "map1.dat" i 0 u 1:2:3


reset
set output "map2.png"
set title "nx = ny = 100, eps1 = eps2 = 1.0"
set xlabel "x"
set ylabel "y"
set pm3d map
set palette defined (-10 "blue", 0 "white", 10 "red")
set size ratio -1
splot [0:10][0:10] "map2.dat" i 0 u 1:2:3


reset
set output "map3.png"
set title "nx = ny = 200, eps1 = eps2 = 1.0"
set xlabel "x"
set ylabel "y"
set pm3d map
set palette defined (-10 "blue", 0 "white", 10 "red")
set size ratio -1
splot [0:20][0:20] "map3.dat" i 0 u 1:2:3


reset
set output "map4.png"
set title "nx = ny = 100, eps1 = eps2 = 1.0"
set xlabel "x"
set ylabel "y"
set pm3d map
set palette defined (-1 "blue", 0 "white", 1 "red")
set size ratio -1
splot [0:10][0:10][-0.8:0.8] "map4.dat" i 0 u 1:2:3


reset
set output "map5.png"
set title "nx = ny = 100, eps1 = 1.0, eps2 = 2.0"
set xlabel "x"
set ylabel "y"
set pm3d map
set palette defined (-1 "blue", 0 "white", 1 "red")
set size ratio -1
set cbrange [-0.8:0.8]
splot [0:10][0:10][-0.8:0.8] "map5.dat" i 0 u 1:2:3


reset
set output "map6.png"
set title "nx = ny = 100, eps1 = 1.0, eps2 = 10.0"
set xlabel "x"
set ylabel "y"
set pm3d map
set palette defined (-1 "blue", 0 "white", 1 "red")
set size ratio -1
set cbrange [-0.8:0.8]
splot [0:10][0:10][-0.8:0.8] "map6.dat" i 0 u 1:2:3