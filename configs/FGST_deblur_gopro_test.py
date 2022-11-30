exp_name = 'FGST_gopro'

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='FGST',
        dim=32,
        patch_test=True,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth',
        ),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))

# model training and testing settings
train_cfg = dict(fix_iter=10000)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)

# dataset settings
train_dataset_type = 'SRFolderMultipleGTDataset'
val_dataset_type = 'SRFolderMultipleGTDataset'

train_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:06d}.png'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

test_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:06d}.png'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

demo_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:06d}.png'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(
        samples_per_gpu=1, drop_last=True, persistent_workers=False),  # 8 gpus
    val_dataloader=dict(samples_per_gpu=1, persistent_workers=False),
    test_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=1, persistent_workers=False),

    # train
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='data/GoPro/train/blur',
            gt_folder='data/GoPro/train/GT',
            num_input_frames=2,
            pipeline=train_pipeline,
            scale=1,
            ann_file='data/GoPro_train.txt',
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        lq_folder='data/GoPro/test/blur',
        gt_folder='data/GoPro/test/GT',
        pipeline=test_pipeline,
        scale=1,
        ann_file='data/GoPro_test.txt',
        test_mode=True),
    # test
    test=dict(
        type=val_dataset_type,
        lq_folder='data/GoPro/test/blur',
        gt_folder='data/GoPro/test/blur',
        pipeline=test_pipeline,
        scale=1,
        ann_file='data/GoPro_test.txt',
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=0,
        betas=(0.9, 0.99),
        paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)})))

# learning policy
total_iters = 1
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[total_iters],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=1, save_image=True, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./experiments/{exp_name}'
load_from = './pretrained_models/FGST.pth'
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True