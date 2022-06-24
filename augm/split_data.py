from tqdm import tqdm
import random
import shutil
import os

dirpath = "dataset/FaceScrub"
list_actors = os.listdir(dirpath)

for name_actor in tqdm(sorted(list_actors)):
    
    dir_train = "dataset/train/ori/" + name_actor + "/"
    dir_test = "dataset/test/ori/" + name_actor + "/"
    
    if not os.path.exists(dir_train):
        os.makedirs(dir_train)
    if not os.path.exists(dir_test):
        os.makedirs(dir_test)
    
    actor_path = os.listdir(f"{dirpath}/{name_actor}")
    random.shuffle(actor_path)
    actor_path = actor_path[:45]
    
    total_train = int(len(actor_path) * 0.8)
    i = 0
    
    for photo in actor_path:
        dir_photo = dirpath + "/" + name_actor + "/" + photo
        
        if i <= total_train:
            shutil.copy(dir_photo, dir_train)
        else:
            shutil.copy(dir_photo, dir_test)
        i += 1

# for name_actor in sorted(list_actors):
#     print(name_actor, end=' - ')
#     print(len(os.listdir(f"dataset/train/ori/{name_actor}")), end=' ')
#     print(len(os.listdir(f"dataset/test/ori/{name_actor}")))
