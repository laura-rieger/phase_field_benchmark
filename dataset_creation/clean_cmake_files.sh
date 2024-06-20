echo " "
echo "# ================== Cleaning cmake files ========================"
rm -f CMakeCache.txt 
rm -f cmake_install.cmake
rm -f main
rm -f Makefile
rm -f -r build/
rm -f -r CMakeFiles
echo "# ================== Cleaning completed ========================"
echo " "
cmake CMakeLists.txt
