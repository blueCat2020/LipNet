gpu = '0'
random_seed = 0
video_path = 'F:/Video/hack_data/1_train/hack_lip_train'
txt_list = f'data/hack_lip_train.txt'
train_list = f'data/hackLip_train.txt'
val_list = f'data/hackLip_val.txt'
batch_size = 32
base_lr = 2e-5
num_workers = 4
max_epoch = 1024
max_frame_len=14
display = 8
test_step = 128
save_prefix = f'weights/LipNet'
is_optimize = True

weights = 'weights/LipNet_loss_2.858724594116211_cer_1.0.pt'
