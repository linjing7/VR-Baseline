import os
import argparse

def recollect(path, new_path):
    path = os.path.join(path, 'quantitative_datasets')
    new_path = os.path.join(new_path, 'quantitative_datasets')
    for item in os.listdir(path):
        # print(item)
        old_path_gt = os.path.join(path,item,'GT', '*.jpg')
        old_path_input = os.path.join(path, item, 'input', '*.jpg')
        new_path_gt = os.path.join(new_path,'GT', item,)
        new_path_input = os.path.join(new_path, 'LQ', item)
        os.makedirs(new_path_gt, exist_ok=True)
        os.makedirs(new_path_input, exist_ok=True)
        os.system(f'cp -r {old_path_gt} {new_path_gt}')
        os.system(f'cp -r {old_path_input} {new_path_input}')

def rename(path):
    path = os.path.join(path, 'quantitative_datasets')
    for type in ['LQ', 'GT']:
        path_ = os.path.join(path, type)
        for item in os.listdir(path_):
            folder_path = os.path.join(path_, item)
            imgs = sorted(os.listdir(folder_path))
            for i in range(len(imgs)):
                os.rename(os.path.join(folder_path, imgs[i]), os.path.join(folder_path, f'{i:05d}.jpg'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data preparation')
    parser.add_argument(
        '--input_path',
        default='data/DeepVideoDeblurring_Dataset/',
        type=str,
        help='path of the original DVD dataset')
    parser.add_argument(
        '--save_path',
        default='data/DVD/',
        type=str,
        help='path of the recollected DVD dataset')
    args = parser.parse_args()
    origin_path = args.input_path
    save_path = args.save_path
    # recollect(origin_path, save_path)
    rename(save_path)