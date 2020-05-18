GARBFNX_FS : global.o file_man.o fitness.o RBFX_sup.o selection.o statistics.o transformation.o util_functions.o
	g++ -Wall global.o file_man.o fitness.o RBFX_sup.o selection.o statistics.o transformation.o util_functions.o -o GARBFNX_FS

global.o : global.cpp	
	g++ -Wall -o global.o -c global.cpp

file_man.o : file_man.cpp	
	g++ -Wall -o file_man.o -c file_man.cpp

fitness.o : fitness.cpp	
	g++ -Wall -o fitness.o -c fitness.cpp

RBFX_sup.o : RBFX_sup.cpp	
	g++ -Wall -o RBFX_sup.o -c RBFX_sup.cpp

selection.o : selection.cpp	
	g++ -Wall -o selection.o -c selection.cpp

statistics.o : statistics.cpp	
	g++ -Wall -o statistics.o -c statistics.cpp

transformation.o : transformation.cpp	
	g++ -Wall -o transformation.o -c transformation.cpp

util_functions.o : util_functions.cpp	
	g++ -Wall -o util_functions.o -c util_functions.cpp

