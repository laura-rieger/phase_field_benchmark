1.) replace my "CMakeLists.txt" with your own which was generated, when you installed the SUNDIALS IDA library from the examples folder
2.)change the following lines in CMakeLists.txt with the following code:

	# Set additional libraries
	set(SUNDIALS_EXTRA_LIBS  -lm -lrt CACHE STRING "Additional libraries")

	set(CMAKE_C_FLAGS
 	"-g -Wall -O3"
 	CACHE STRING "C compiler flags")


	# Set the names of the examples to be built and their dependencies
	set(examples main) 


3.) run the following command to prepare the libraries and produce the Makefile:

	cmake CMakeLists.txt

4.) run the following command to compile the program

	make

5.) run the following command to run the program

	./main

6.) you should get a terminal output that looks something like this

Spinodal decomposition with Cahn-Hilliard 2D DAE serial problem for IDA developed by University of Ljubljana
Grid size: 50 x 50
Linear solver: SPGMR, preconditioner using diagonal elements.
Tolerance parameters:  rtol = 1e-10   atol = 1e-10 1e-08 1e-10 
Initial conditions y0 = (0.499265 0.0036445 0.495443)
Constraints and id not used.

-----------------------------------------------------------------------
  t             y1           y2           y3      | nst  k      h
-----------------------------------------------------------------------
4.0040e-02   4.9447e-01   1.6802e-03   4.9436e-01 | 629  4   2.9119e-04

Final Run Statistics: 

Number of steps                    = 629
Number of residual evaluations     = 3514
Number of Jacobian evaluations     = 0
Number of nonlinear iterations     = 696
Number of error test failures      = 0
Number of nonlinear conv. failures = 0
Number of root fn. evaluations     = 0

7.) for running multiple cases, I have included "runSerial.sh" script