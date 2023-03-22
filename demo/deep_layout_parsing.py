
# https://layout-parser.readthedocs.io/en/latest/example/deep_layout_parsing/index.html

import layoutparser as lp
import cv2
import os
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"E:\Program Files\Tesseract-OCR\tesseract.exe"

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

    ###############################################
    # 过滤文本类型区块
    text_blocks = lp.Layout([b for b in layout if b.type=='Text'])
    figure_blocks = lp.Layout([b for b in layout if b.type=='Figure'])
    text_blocks = lp.Layout([b for b in text_blocks \
                   if not any(b.is_in(b_fig) for b_fig in figure_blocks)])
    
    # 按左右分区
    h, w = image.shape[:2]
    # https://layout-parser.readthedocs.io/en/latest/api_doc/elements.html#layoutparser.elements.Interval
    left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image)
    # put_on_canvas 后 area canvas_width canvas_height 轴线上的距离会被赋值
    # 相当于在应用场景上实例化

    left_blocks = text_blocks.filter_by(left_interval, center=True)
    left_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)

    right_blocks = lp.Layout([b for b in text_blocks if b not in left_blocks])
    right_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)

    # And finally combine the two list and add the index
    # according to the order
    text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])

    im_show = lp.draw_box(image, text_blocks, box_width=3)
        # Show the detected layout of the input image
    # im_show = Image.fromarray(im_show)
    im_show.save(os.path.join(save_folder,imgFileName,'result2.jpg'))

    ###############################
    # 获取区域文本
    ocr_agent = lp.TesseractAgent(languages='eng')
    # Initialize the tesseract ocr engine. You might need
    # to install the OCR components in layoutparser:
    # pip install layoutparser[ocr]
    for block in text_blocks:
        segment_image = (block
                       .pad(left=5, right=5, top=5, bottom=5)
                       .crop_image(image))
        # add padding in each image segment can help
        # improve robustness

        text = ocr_agent.detect(segment_image)
        block.set(text=text, inplace=True)
    for txt in text_blocks.get_texts():
        print(txt, end='\n---\n')
layoutPatserFun()
