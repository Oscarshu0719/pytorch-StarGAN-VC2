# VCC 2016.

#!/bin/bash

spk_select_train_dir="./data/spk"
spk_select_test_dir="./data/spk_test" 

if [ ! -d ${spk_select_train_dir} ]; then
	mkdir ${spk_select_train_dir}
fi

if [ ! -d ${spk_select_test_dir} ]; then
	mkdir ${spk_select_test_dir}
fi

for var in "$@"
do
	spk_train_dir="./data/vcc2016_training/${var}"
	spk_test_dir="./data/evaluation_all/${var}"

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
