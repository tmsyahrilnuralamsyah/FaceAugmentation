from mtcnn import MTCNN
from imutils import paths
from tqdm import tqdm
import cv2
import os

detector = MTCNN()
dirpath = "dataset/train/yaw+pitch+roll"
imagePaths = sorted(list(paths.list_images(dirpath)))

for imagePath in tqdm(imagePaths):
    path_split = imagePath.split(os.sep)
    name_actor = path_split[-2]
    fn = path_split[-1]
    fn = fn.split('.')
    filename = fn[0]
    fileformat = fn[1]
    
    dirdest = "dataset/train/mtcnn/yaw+pitch+roll/" + name_actor + "/"
    
    if not os.path.exists(dirdest):
        os.makedirs(dirdest)

    image = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)

    for i in range(len(result)):
        bounding_box = result[i]['box']
        keypoints = result[i]['keypoints']

        bounding_box[0] = 0 if bounding_box[0] < 0 else bounding_box[0]
        bounding_box[1] = 0 if bounding_box[1] < 0 else bounding_box[1]

        path_save = dirdest + filename + "." + fileformat
        img = image[bounding_box[1]:bounding_box[1] + bounding_box[3], \
            bounding_box[0]:bounding_box[0] + bounding_box[2]]
        cv2.imwrite(path_save, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
