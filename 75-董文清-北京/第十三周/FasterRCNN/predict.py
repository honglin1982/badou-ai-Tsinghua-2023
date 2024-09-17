from PIL import  Image
from frcnn import FRCNN

frcnn = FRCNN()
while True :

    path = 'img/street.jpg'
    img = input(path)
    try:
        image = Image.open(path)
    except :
        print('Open Error, Try again...')
        continue
    else :
        rImage = frcnn.detectImage(image)
        rImage.show()
frcnn.closeSession()
