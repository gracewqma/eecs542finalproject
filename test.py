from augmentations import hue, saturation, contrast

from matplotlib import image



img_path = "dataset/obama/100-shot-obama/0.jpg"

img = image.imread(img_path)
hue(img)
saturation(img)
contrast(img)