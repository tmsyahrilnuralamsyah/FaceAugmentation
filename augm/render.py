from image_transformer import ImageTransformer
from util import save_image
from imutils import paths
from tqdm import tqdm
import os

# Parameters:
#     image_path: the path of image that you want rotated
#     shape     : the ideal shape of input image, None for original size.
#     theta     : rotation around the x axis
#     phi       : rotation around the y axis
#     gamma     : rotation around the z axis
#     dx        : translation along the x axis
#     dy        : translation along the y axis
#     dz        : translation along the z axis

dirpath = "dataset/train/ori"
imagePaths = sorted(list(paths.list_images(dirpath)))
rot_range = 20
img_shape = None

for imagePath in tqdm(imagePaths):
    path_split = imagePath.split(os.sep)
    name_actor = path_split[-2]
    fn = path_split[-1]
    fn = fn.split('.')
    filename = fn[0]
    fileformat = fn[1]
    
    dirdest = "dataset/train/yaw+pitch+roll/" + name_actor + "/"
    
    if not os.path.exists(dirdest):
        os.makedirs(dirdest)
    
    img = ImageTransformer(imagePath, img_shape)
    
    for ang in range(rot_range+1):
        
        if ang == 0:
            rotated_img = img.rotate_along_axis()
            save_image(dirdest + filename + '_original.jpg', rotated_img)
        
        # yaw
        if ang%10 == 0 and ang <= 20 and ang != 0:
            rotated_img = img.rotate_along_axis(phi=ang)
            save_image(dirdest + filename + '_yaw+{}.jpg'.format(str(ang)), rotated_img)
            rotated_img = img.rotate_along_axis(phi=-ang)
            save_image(dirdest + filename + '_yaw-{}.jpg'.format(str(ang)), rotated_img)

        # picth
        if ang%10 == 0 and ang <= 20 and ang != 0:
            rotated_img = img.rotate_along_axis(theta=ang)
            save_image(dirdest + filename + '_picth+{}.jpg'.format(str(ang)), rotated_img)
            rotated_img = img.rotate_along_axis(theta=-ang)
            save_image(dirdest + filename + '_picth-{}.jpg'.format(str(ang)), rotated_img)
        
        # roll
        if ang%10 == 0 and ang <= 10 and ang != 0:
            rotated_img = img.rotate_along_axis(gamma=ang)
            save_image(dirdest + filename + '_roll+{}.jpg'.format(str(ang)), rotated_img)
            rotated_img = img.rotate_along_axis(gamma=-ang)
            save_image(dirdest + filename + '_roll-{}.jpg'.format(str(ang)), rotated_img)
