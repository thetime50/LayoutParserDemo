
# https://layout-parser.readthedocs.io/en/latest/example/deep_layout_parsing/index.html

import layoutparser as lp
import cv2
import os
from PIL import Image

dir = os.path.dirname(__file__)
resDir = os.path.join(dir,'result/start')

def layoutPatserFun():
    print('layoutPatserFun')
    save_folder = os.path.join(resDir,'layoutPatserFun')
    imgFileName = 'paper-image'
    image = cv2.imread(os.path.join(dir,"../../layout-parser/examples/data/paper-image.jpg"))
    image = image[..., ::-1]
        # Convert the image from BGR (cv2 default loading style)
        # to RGB


    model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                    label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
        # Load the deep layout model from the layoutparser API
        # For all the supported model, please check the Model
        # Zoo Page: https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html

    layout = model.detect(image)
        # Detect the layout of the input image

    im_show = lp.draw_box(image, layout, box_width=3)
        # Show the detected layout of the input image
    # im_show = Image.fromarray(im_show)
    im_show.save(os.path.join(save_folder,imgFileName,'result.jpg'))

layoutPatserFun()
