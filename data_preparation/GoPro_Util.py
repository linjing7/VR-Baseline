import os

def recollect(path, new_path):
    for item in os.listdir(path):
        print(item)
        old_path_gt = os.path.join(path,item,'sharp', '*')
        old_path_input = os.path.join(path, item, 'blur', '*')
        new_path_gt = os.path.join(new_path,'GT', item)
        new_path_input = os.path.join(new_path, 'blur', item)
        if not os.path.exists(new_path_gt):
            os.makedirs(new_path_gt)
            os.makedirs(new_path_input)
        os.system(f'cp -r {old_path_gt} {new_path_gt}')
        os.system(f'cp -r {old_path_input} {new_path_input}')

def rename(path):
    for item in os.listdir(path):
        folder_path = os.path.join(path, item)
        imgs = sorted(os.listdir(folder_path))
        for i in range(len(imgs)):
            os.rename(os.path.join(folder_path, imgs[i]), os.path.join(folder_path, f'{i:06d}.png'))
            print(imgs[i], f'{i:06d}.png')

if __name__ == '__main__':
    origin_path = '/data/gopro_ds/'
    new_path = '/data/GoPro/'
    for type1 in ['train', 'test']:
        recollect(os.path.join(origin_path, type1), os.path.join(new_path, type1))
        for type2 in ['blur', 'GT']:
            rename(os.path.join(new_path, type1, type2))