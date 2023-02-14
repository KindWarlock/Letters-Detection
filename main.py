import cv2
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from tqdm import tqdm
from skimage.measure import regionprops, label


def extract_features(image):
    if image.ndim == 3:
        gray = np.mean(image, 2)
        gray[gray > 0] = 1
        labeled = label(gray)
    else:
        labeled = image.astype('uint8')
    props = regionprops(labeled)[0]    
    extent = props.extent
    eccentricity = props.eccentricity
    euler = props.euler_number
    rr, cc = props.centroid_local
    rr = rr / props.image.shape[0]
    cc = cc / props.image.shape[1]
    feret = (props.feret_diameter_max - 1) / np.max(props.image.shape)    
    return np.array([extent, eccentricity, euler, rr, cc, feret], dtype='f4')


text_images = [plt.imread(path) for path in pathlib.Path('./out').glob('*.png')]

train_images = {}
for path in tqdm(sorted(pathlib.Path('out/train').glob('*'))):
    symbol = path.name[-1]
    train_images[symbol] = []
    for image_path in sorted(path.glob('*.png')):
        train_images[symbol].append(plt.imread(image_path))

knn = cv2.ml.KNearest_create()
train = []
responses = []

sym2class = {symbol: i for i, symbol in enumerate(train_images)}
class2sym = {value: key for key, value in sym2class.items()}

for symbol in tqdm(train_images):
    for image in train_images[symbol]:
        train.append(extract_features(image))
        responses.append(sym2class[symbol])

train = np.array(train, dtype='f4')
responses = np.array(responses).reshape(-1, 1).astype('f4')

knn.train(train, cv2.ml.ROW_SAMPLE, responses)


def image2text(image) -> str:
    gray = np.mean(image, 2)
    gray[gray > 0] = 1
    labeled = label(gray)
    regions = regionprops(labeled)
    regions.sort(key = lambda x: x.bbox[1])
    
    # направление роста
    ascending = regions[0].bbox[0] < regions[len(regions) - 1].bbox[0]
    
    # найти большие отклонения от среднего расстояния между буквами - пробелы (больше) и i (меньше)
    prev = 0
    dist = []
    for region in regions:
        dist.append(region.bbox[1] - prev)
        prev = region.bbox[3]
    avg = np.average(dist)
    std = np.std(dist)

    prev = 0
    answer = []
    for region in regions:
        if region.bbox[1] - prev > (avg + std):
            answer.append(' ')

        # обработка i: если фраза растет, то сначала по порядку идет точка (ее мы убираем), 
        # иначе - длинная часть (тогда точку пропускаем)
        elif region.bbox[1] - prev < (avg - std):
            if not ascending:
                continue
            else:
               answer.pop() 
        
        features = extract_features(region.image).reshape(1, -1)
        ret, results, neighbours, dist = knn.findNearest(features, 5)
        answer.append(class2sym[int(ret)])
        prev = region.bbox[3]
    return "".join(answer)

for i in text_images:
    print(image2text(i))

