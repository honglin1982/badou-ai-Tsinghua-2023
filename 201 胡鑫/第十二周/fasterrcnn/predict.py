from PIL import Image
from frcnn import FRCNN

frcnn = FRCNN()
while True:
    try:
        image = Image.open('img/1.jpg')
    except Exception as e:
        print('Open Error! Try again!')
        continue
    else:
        r_image = frcnn.detect_image(image)
        r_image.show()
frcnn.close_session()
