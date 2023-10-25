python main.py \
    --dataset CIFAR-100 \
    --data_folder /home/ycg/old_disk/user/dataset/ \
    --kd-method CIMGS-KD \
    --checkpoint-dir ./checkpoint/

python main.py \
    --dataset CIFAR-100 \
    --data_folder /home/ycg/old_disk/user/dataset/ \
    --kd-method cross_entropy \
    --checkpoint-dir ./checkpoint/



python main.py \
    --arch resnet18 \
    --dataset CUB200 \
    --data_folder /home/ycg/old_disk/user/dataset/CUB_200_2011/ \
    --kd-method CIMGS-KD \
    --checkpoint-dir ./checkpoint/

python main.py \
    --arch resnet18 \
    --dataset CUB200 \
    --data_folder /home/ycg/old_disk/user/dataset/CUB_200_2011/ \
    --kd-method cross_entropy \
    --checkpoint-dir ./checkpoint/

python main.py \
    --arch resnet18 \
    --dataset STANFORD120 \
    --data_folder /home/ycg/old_disk/user/dataset/standford-dogs \
    --kd-method cross_entropy \
    --checkpoint-dir ./checkpoint/

python main.py \
    --arch resnet18 \
    --dataset STANFORD120 \
    --data_folder /home/ycg/old_disk/user/dataset/standford-dogs \
    --kd-method CIMGS-KD \
    --checkpoint-dir ./checkpoint/


python main.py \
    --arch resnet18 \
    --dataset imagenet \
    --data_folder /home/ycg/old_disk/user/dataset/ImageNet \
    --kd-method CIMGS-KD \
    --checkpoint-dir ./checkpoint/

python main.py \
    --arch resnet18 \
    --dataset imagenet \
    --data_folder /home/ycg/old_disk/user/dataset/ImageNet \
    --kd-method cross_entropy \
    --checkpoint-dir ./checkpoint/
