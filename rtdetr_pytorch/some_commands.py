train:
CUDA_VISIBLE_DEVICES=9 torchrun --master_port=9909 --nproc_per_node=1 tools/train.py -c /data1/zy/code/rt_detr/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml -t weight/rtdetr_r50vd_6x_coco_from_paddle.pth --seed=0

validation: 
CUDA_VISIBLE_DEVICES=9 torchrun --master_port=9909 --nproc_per_node=1 tools/train.py -c /data1/zy/code/rt_detr/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r /data1/zy/code/rt_detr/rtdetr_pytorch/output/rtdetr_r50vd_6x_coco_copper_24e_1/checkpoint0020.pth --test-only --seed=0

inference code to enable folder image visualization:
(rtdetr) zy@NIPS:/data1/zy/code/rt_detr/rtdetr_pytorch$ python tools/torch_inf.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r output/rtdetr_r50vd_6x_coco_copper_24e_1/checkpoint0020.pth --input /data1/zy/dataset/cooper001/newsplit/coco/test2017/ --device cuda:0 --output test_vis

ps. if the ids of target classes start from 1 instead of 0, the rectification should be made the same as that of the matcher.py in D-FINE.
