node 1 0. 0.
node 2 1.5 0.
node 3 3. 0.
node 4 4.5 0.
node 5 6. 0.
node 6 7.5 0.
node 7 9. 0.
node 8 10.5 0.
node 9 12. 0.
node 10 0. 1.5
node 11 1.5 1.5
node 12 3. 1.5
node 13 4.5 1.5
node 14 6. 1.5
node 15 7.5 1.5
node 16 9. 1.5
node 17 10.5 1.5
node 18 12. 1.5
node 19 0. 3.
node 20 1.5 3.
node 21 3. 3.
node 22 4.5 3.
node 23 6. 3.
node 24 7.5 3.
node 25 9. 3.
node 26 10.5 3.
node 27 12. 3.

node 28 3 4

element CP4 1 1 2 11 10 1 .2
element CP4 2 2 3 12 11 1 .2
element CP4 3 3 4 13 12 1 .2
element CP4 4 4 5 14 13 1 .2
element CP4 5 5 6 15 14 1 .2
element CP4 6 6 7 16 15 1 .2
element CP4 7 7 8 17 16 1 .2
element CP4 8 8 9 18 17 1 .2
element CP4 9 10 11 20 19 1 .2
element CP4 10 11 12 21 20 1 .2
element CP4 11 12 13 22 21 1 .2
element CP4 12 13 14 23 22 1 .2
element CP4 13 14 15 24 23 1 .2
element CP4 14 15 16 25 24 1 .2
element CP4 15 16 17 26 25 1 .2
element CP4 16 17 18 27 26 1 .2

material Elastic1D 4 1E6

element T2D2 20 21 28 4 1

fix 1 P 1 10 19 28

material PlaneStress 1 2
material Elastic3D 2 3E4 .0 .75E-1

amplitude Tabular 1 motion_time
# amplitude Tabular 1 up_motion_time
acceleration 2 1 9.81 2

modifier Rayleigh 1 0 .002 0. 0. 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 20

hdf5recorder 1 Node U 2 3 4 5 6 7 8 9 11 12 13 14 15 16 17 18 20 21 22 23 24 25 26 27 28
hdf5recorder 2 Node DF 2 3 4 5 6 7 8 9 11 12 13 14 15 16 17 18 20 21 22 23 24 25 26 27 28
hdf5recorder 3 Node IF 2 3 4 5 6 7 8 9 11 12 13 14 15 16 17 18 20 21 22 23 24 25 26 27 28

# step frequency 1 50
# analyze
# peek eigenvalue

step dynamic 1 60
set ini_step_size 4E-3
set fixed_step_size true

converger AbsIncreDisp 1 1E-12 10 false

integrator Newmark 1

analyze

save recorder 1 2 3

exit