L=$1

# make new folder
foldername="train_L${L}"
mkdir ${foldername}
cp main.py run_python_gpu utils.py ${foldername}
cd ${foldername}
sbatch run_python_gpu main.py