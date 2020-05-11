# Support VCC2016 and VCC2018 two datasets.

#!/bin/bash

if [ $# -lt 2 ]; then
	echo "Only $# args are found."
	exit
fi

if [ ${1^^} = "VCC2016" ]; then
	root_train_dir="./data/vcc2016_training"
	root_test_dir="./data/evaluation_all"
elif [ ${1^^} = "VCC2018" ]; then
	root_train_dir="./data/vcc2018_database_training"
	root_test_dir="./data/vcc2018_database_evaluation"
else
	echo "${1} is NOT supported."
	exit
fi

spk_select_train_dir="./data/spk"
spk_select_test_dir="./data/spk_test" 

if [ ! -d ${spk_select_train_dir} ]; then
	mkdir ${spk_select_train_dir}
fi

if [ ! -d ${spk_select_test_dir} ]; then
	mkdir ${spk_select_test_dir}
fi

# Skip first arg.
shift

for var in "$@"
do
	spk_train_dir="${root_train_dir}/${var}"
	spk_test_dir="${root_test_dir}/${var}"

	if [ -d ${spk_train_dir} ]; then
		mv ${spk_train_dir} ${spk_select_train_dir}
	else
		echo "${spk_train_dir} does NOT exist."
	fi

	if [ -d ${spk_test_dir} ]; then
		mv ${spk_test_dir} ${spk_select_test_dir}
	else
		echo "${spk_test_dir} does NOT exist."
	fi
done
