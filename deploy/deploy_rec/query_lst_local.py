import sys,os
import urllib2
import json
import cv2
import numpy as np
import random
import time
from PIL import Image, ImageEnhance, ImageDraw, ImageFont 

#address="http://192.168.6.47:8089/upload"
#address="http://10.57.239.242:8089/upload"

from ocr import OCR
words_detect=OCR()

def imread_binary(img_buffer):
    img = np.asarray(bytearray(img_buffer), dtype='uint8')
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if len(img.shape) == 2 :
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img

def draw_bbox(img, bboxes, color):
    bboxes=[bboxes]
    font = ImageFont.truetype('simkai.ttf', 14)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        if 'x1' in bbox:
            x1=bbox['x1']
            y1=bbox['y1']
            x2=bbox['x2']
            y2=bbox['y2']
            x3=bbox['x3']
            y3=bbox['y3']
            x4=bbox['x4']
            y4=bbox['y4']
            text = bbox['text']
            #cv2.line(img,(int(x1),int(y1)),(int(x1),int(y1)),color,2)
            #cv2.line(img,(int(x2),int(y2)),(int(x2),int(y2)),color,2)
            #cv2.line(img,(int(x3),int(y3)),(int(x3),int(y3)),color,2)
            #cv2.line(img,(int(x4),int(y4)),(int(x4),int(y4)),color,2)
            draw.line((int(x1),int(y1),int(x1),int(y1)), color, 2)
            draw.line((int(x2),int(y2),int(x2),int(y2)), color, 2)
            draw.line((int(x3),int(y3),int(x3),int(y3)), color, 2)
            draw.line((int(x4),int(y4),int(x4),int(y4)), color, 2)
            draw.text((x1+10,max(y1-6,0)), text, (0,200,0), font=font)
        else:
            xmin = bbox['xmin']
            ymin = bbox['ymin']
            xmax = bbox['xmax']
            ymax = bbox['ymax']
        #img = np.array(img)

    return np.array(img)

def img_crop(img, xmin, ymin, xmax, ymax, offset_w=0, offset_h=0):
    [height, width, c] = img.shape
    xmin_c = xmin - offset_w if xmin - offset_w > 0 else 0
    ymin_c = ymin - offset_h if ymin - offset_h > 0 else 0
    xmax_c = xmax + offset_w if xmax + offset_w < width else width - 1
    ymax_c = ymax + offset_h if ymax + offset_h < height else height - 1
    return img[ymin_c:ymax_c+1, xmin_c:xmax_c+1, :]

# write detection result as txt files
def write_result_in_one_txt(image_data, det_file, image_name, bboxes, crop_img_save=None):
    for b_idx, bbox in enumerate(bboxes):
        xmin,ymin,xmax,ymax = 0,0,0,0
        if 'x1' in bbox:
            line = "%d, %d, %d, %d, %d, %d, %d, %d"%(bbox['x1'],bbox['y1'],\
                    bbox['x2'],bbox['y2'],bbox['x3'],bbox['y3'],bbox['x4'],bbox['y4'])
            xmin = min(bbox['x1'],bbox['x2'],bbox['x3'],bbox['x4'])
            ymin = min(bbox['y1'],bbox['y2'],bbox['y3'],bbox['y4'])
            xmax = max(bbox['x1'],bbox['x2'],bbox['x3'],bbox['x4'])
            ymax = max(bbox['y1'],bbox['y2'],bbox['y3'],bbox['y4'])
            text = bbox['text']
        else:
            xmin = bbox['xmin'] 
            ymin = bbox['ymin'] 
            xmax = bbox['xmax'] 
            ymax = bbox['ymax'] 
            text = bbox['text']
        #det_file.write("%s, %d, %d, %d, %d, %d, %d, %d, %d\n"%(image_name,tuple(bbox)))
        det_file.write("{} {} {} {} {} {}\n".format(image_name,text.encode('utf-8'),xmin,ymin,xmax,ymax))
        if crop_img_save is not None:
            crop_name = crop_img_save + '_crop{}.png'.format(b_idx)
            img_crop_data=img_crop(image_data, xmin, ymin, xmax, ymax)
            #img_water_idx= np.where(img_crop_data>235)
            #img_gray_idx= np.where(img_crop_data<180)
            #img_crop_data[img_water_idx]=255
            #img_crop_data[img_gray_idx]=0
            #cv2.imwrite(crop_name, img_crop_data)
        image_data = draw_bbox(image_data, bbox, color = (0, 0, 255))
    return image_data

if __name__ == '__main__':
    imglst=sys.argv[1]
    imgout=sys.argv[2]
    det_output = open(imgout + '.csv', 'w')
    det_output_imshow = imgout + '.imshow'
    if not os.path.exists(det_output_imshow):
        os.popen('mkdir -p %s'%det_output_imshow)

    index=0
    with open(imglst) as f:
        lines = f.readlines()
        total_num = len(lines)
        for line in lines:
        #line = random.choice(lines)
        #while line:
        #    line = random.choice(lines)
            img_dir = line.strip()
            index += 1
            img_data=open(line.strip(), mode='rb')
            img_data_size=os.path.getsize(line.strip())
            img_name=line.strip().split('/')[-1]
            #img_folder=line.strip().split('/')[-2]
           
            bboxes=[]
            res={}
            result=''
            time_start = 0
            time_end = 0
            if not "service":
                request=urllib2.Request(address, data=img_data)
                request.add_header('Content-Length', '%d'%img_data_size)

                response=urllib2.urlopen(request)
                result=response.read()
            if "local":
                time_start = time.time()
                result=words_detect.rec_warper(img_data.read())
                time_end = time.time()
            print result
            continue
            res=json.loads(result)
            bboxes=res['textline']
            #print bboxes 
            print "{}/{} time:{:.4f} {:.2f} {}".format(index, total_num, time_end - time_start, index/float(total_num), img_dir)
            
            # For debug.
            if int(res['textline_num']) > 0 :
                img_data=open(line.strip(), mode='rb')
                image_data=imread_binary(img_data.read())
                #image_show=write_result_in_one_txt(image_data, det_output, img_name, bboxes)
                image_show=write_result_in_one_txt(image_data, det_output, img_name, bboxes, det_output_imshow + '/'  + img_name)
                cv2.imwrite(det_output_imshow + '/'  + img_name, image_show)
            img_data.close()
    det_output.close()
