#!/bin/bash

# read input parameters for restart procedure
# usage with the console arguments: ./runSerial -r yes -d Results/Job_2021XXXX_XX/
while getopts r:d: flag
do
    case "${flag}" in
        r) restart=${OPTARG};;
        d) jobDirRestart=${OPTARG};;
    esac
done

# number of cases with different random distributions of concentration
Ncase_Crand=3

# arrays of dimensional parameters to be varied
meanC=(9120 11400) 	#array for dimensional mean values of concentration
RPC=(228 456) 		#array for dimensional values of random perturbation of concentration 

# determine size of declared arrays
size_meanC=${#meanC[@]}
size_RPC=${#RPC[@]}

# calculate total number of cases
totalCases=$(( $Ncase_Crand*$size_meanC*$size_RPC ))

# root folder
rootFolder="Results"

#default folder for storing results
defaultResultFolder="default"

# folder for storing various logs
logsFolder="logs"

# prefix name for the work folder name
workFolder='Job'

# prefix name for the individual case folder name
caseFolder='Case'

# simulation output file
simOutput='simulation.out'

#create directory /Results if it does not exist
mkdir -p $rootFolder

#create default directory for storing results /Results/default if it does not exist
mkdir -p $rootFolder'/'$defaultResultFolder

#create directory for storing logs /Results/logs if it does not exist
mkdir -p $rootFolder'/'$logsFolder

# construct string with date
Year=`date +%Y`
Month=`date +%m`
Day=`date +%d`
dateString=$Year$Month$Day

#check if index file already exists, if not create one and write 0 to it
indexFile=index.txt
indexDir=$rootFolder'/'$logsFolder'/'$indexFile
if [ -f "$indexDir" ]; then
echo 0 > /dev/null
else
echo 0 > $indexDir
fi

# read job index from index.txt file, increase it by 1 and store it back to the file 
jobIndex=$(< $indexDir)
jobIndex=$(($jobIndex + 1))
echo $jobIndex > $indexDir


#check if history log file already exists, if not create one and write header to it
historyLog=history.csv
historyDir=$rootFolder'/'$logsFolder'/'$historyLog

if [ -f "$historyDir" ]; then
echo 0 > /dev/null
else
echo 'Date;Time;Job;Case;Mean C;Perturbation C;Index of randC' > $historyDir
fi

# depending on the restart condition define jobName and jobDirectory and print information to console 
if [ "$restart" = "yes" ]; then
    jobName=restart
	jobDirectory=$jobDirRestart
	echo ================== RESTART ==================
	echo Current restart job directory: $jobDirectory
	echo Total number of restarted cases: $totalCases
else
    jobName=$workFolder'_'$dateString'_'$jobIndex
	jobDirectory=$rootFolder'/'$jobName'/'
	mkdir $jobDirectory
	echo ================== START ==================
	echo Current job directory: $jobDirectory
	echo Total number of cases: $totalCases
fi



# counter
tmp=1
echo ''

# compile the code
make
echo ''


# for loops that execute simulation for every combination of the input parameters
for (( i_meanC=0; i_meanC<$size_meanC; i_meanC++ ))
do
	for (( i_RPC=0; i_RPC<$size_RPC; i_RPC++ ))
	do
		for (( i_cRand=1; i_cRand<=$Ncase_Crand; i_cRand++ ))
		do
		
		caseDirectory=$jobDirectory$caseFolder'_'${meanC[$i_meanC]}'_'${RPC[$i_RPC]}'_'$i_cRand'/'		
		mkdir -p $caseDirectory
		
		#run sequentially different cases based on the restart condition
		if [ "$restart" = "yes" ]; then
			echo Restarting $caseFolder'_'${meanC[$i_meanC]}'_'${RPC[$i_RPC]}'_'$i_cRand [$tmp/$totalCases]
			time ./main $i_cRand ${meanC[$i_meanC]} ${RPC[$i_RPC]} ./$caseDirectory >> $caseDirectory$simOutput 2>&1
			
		else
			echo Running $caseFolder'_'${meanC[$i_meanC]}'_'${RPC[$i_RPC]}'_'$i_cRand [$tmp/$totalCases]
			time ./main $i_cRand ${meanC[$i_meanC]} ${RPC[$i_RPC]} ./$caseDirectory >> $caseDirectory$simOutput 2>&1
		fi

		#save information to the history log
		Time=`date +"%T"`
		echo $Year-$Month-$Day';'$Time';'$jobName';'$caseFolder'_'${meanC[$i_meanC]}'_'${RPC[$i_RPC]}'_'$i_cRand';'${meanC[$i_meanC]}';'${RPC[$i_RPC]}';'$i_cRand >> $historyDir
		echo ''
		echo ''
		
		#increase counter
		let tmp++
		
		done 
	done
done
