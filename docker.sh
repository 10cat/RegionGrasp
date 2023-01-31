docker run -it --rm  --gpus all -v /home:/home -v /media/data:/media/data -v /mnt/sda:/mnt/sda --shm-size 300G --ipc=host  hoi:v1
