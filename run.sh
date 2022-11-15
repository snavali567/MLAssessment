#!/bin/bash
if [[ $(conda --version) ]]; then
	echo "Conda exists"
else
	echo "Conda does not exist"
	exit 1
fi

# create virtual environment model training

# shellcheck disable=SC2143
if [[ $(conda env list | grep Samarth_assessment) ]]; then
	echo "Environment exists"
else
	echo "Environment does not exist"
	conda create --name Samarth_assessment python=3.8 -u -p

fi

conda activate Samarth_assessment

# install requirements
pip3 install -r requirements.txt

# Check if installations are present in pip list
# shellcheck disable=SC2143
if [[ $(pip list | grep -E 'Flask|pandas|boto|scikit|pickle') ]]; then
	echo "All libraries are present"
else
	echo "Manually install required libraries from requirements.txt"
	exit 1
fi

# run model training
python3 model.py

# run Flask app
echo "To stop the APP use Ctrl + Z"
python3 app.py

