import sys,os
import cv2
import numpy as np
from predict import TextLineOCR
import json
import time

class OCR():
    def __init__(self):
        self.textline_reco = TextLineOCR()
        self.image_data = None
    
    def imread_binary(self, img_buffer):
        img = np.asarray(bytearray(img_buffer), dtype='uint8')
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if img is None:
            return None
        if len(img.shape) == 2 :
            img = img[:, :, np.newaxis]
            if color:
                img = np.tile(img, (1, 1, 3))
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        return img
 
    def img_crop(self, img, xmin, ymin, xmax, ymax, offset_w=0.1, offset_h=0.1):
        [height, width, c] = img.shape
        if offset_w < 1 :
            offset_w = int(offset_w*(xmax - xmin))
        if offset_h < 1 :
            offset_h = int(offset_h*(ymax - ymin))
        xmin_c = xmin - offset_w if xmin - offset_w > 0 else 0
        ymin_c = ymin - offset_h if ymin - offset_h > 0 else 0
        xmax_c = xmax + offset_w if xmax + offset_w < width else width - 1
        ymax_c = ymax + offset_h if ymax + offset_h < height else height - 1
        return img[ymin_c:ymax_c+1, xmin_c:xmax_c+1, :]
    
    def rec_warper(self, crop_img):
        #if self.image_data is None: 
        #    self.image_data = self.imread_binary(crop_img)
        textline_res = self.textline_reco.rec([crop_img])
        res=[]
        for res_ in textline_res:
            #res = [unicode(r, 'utf-8').encode('utf-8') for r in res]
            for r in res_:
                #print type(r)
                if isinstance(r, str):
                    rr = unicode(r, 'utf-8')
                else :
                    rr = r
                res.append(rr.encode('utf-8'))
        ocr_res = ''.join(res)
        #self.image_data = None
        return ocr_res

    def rec_warper_with_dets(self, raw_img, det_res):
        if self.image_data is None: 
            self.image_data = self.imread_binary(raw_img)
            #cv2.imwrite('test.png', self.image_data)
        #det_start = time.time()
        #det_res = self.textline_detector.detect(self.image_data)
        #det_end = time.time()
        ocr_list = []
        if len(det_res) == 0 or self.image_data is None:
            return json.dumps({'textline_num' : len(ocr_list), 'textline' : ocr_list})
        
        #rec_start = time.time()
        for idx, bbox in enumerate(det_res):
            if u'x1' in bbox :
                xmin = min(bbox[u'x1'],bbox[u'x2'],bbox[u'x3'],bbox[u'x4'])
                ymin = min(bbox[u'y1'],bbox[u'y2'],bbox[u'y3'],bbox[u'y4'])
                xmax = max(bbox[u'x1'],bbox[u'x2'],bbox[u'x3'],bbox[u'x4'])
                ymax = max(bbox[u'y1'],bbox[u'y2'],bbox[u'y3'],bbox[u'y4'])
            elif 'xmin' in bbox:
                xmin = bbox['xmin'] 
                ymin = bbox['ymin'] 
                xmax = bbox['xmax'] 
                ymax = bbox['ymax'] 
            else:
                print("Warning : Unrecongnized bbox format!")
                print(bbox)
                continue
            crop_img = self.img_crop(self.image_data, xmin, ymin, xmax, ymax)
            #img_water_idx= np.where(crop_img>235)
            #img_gray_idx= np.where(crop_img<180)
            #crop_img[img_water_idx]=255
            #crop_img[img_gray_idx]=0
            bbox_ocr = bbox
            bbox_ocr['text'] = (self.rec_warper(crop_img)).upper()
            ocr_list.append(bbox_ocr)
        #rec_end = time.time()
        #print("Total time : {}, det time : {}, rec time : {}".format(rec_end - det_start, det_end - det_start, rec_end - rec_start))
        self.image_data = None
        return json.dumps({'textline_num' : len(ocr_list), 'textline' : ocr_list})
        

"""
    def ocr(self, img):
        self.image_data = self.imread_binary(img)
        det_start = time.time()
        det_res = self.textline_detector.detect(self.image_data)
        det_end = time.time()
        ocr_list = []
        if len(det_res) == 0:
            return json.dumps({'textline_num' : len(ocr_list), 'textline' : ocr_list})
        
        rec_start = time.time()
        for idx, bbox in enumerate(det_res):
            if 'x1' in bbox :
                xmin = min(bbox['x1'],bbox['x2'],bbox['x3'],bbox['x4'])
                ymin = min(bbox['y1'],bbox['y2'],bbox['y3'],bbox['y4'])
                xmax = max(bbox['x1'],bbox['x2'],bbox['x3'],bbox['x4'])
                ymax = max(bbox['y1'],bbox['y2'],bbox['y3'],bbox['y4'])
            elif 'xmin' in bbox:
                xmin = bbox['xmin'] 
                ymin = bbox['ymin'] 
                xmax = bbox['xmax'] 
                ymax = bbox['ymax'] 
            else:
                print("Warning : Unrecongnized bbox format!")
                print(bbox)
                break
            crop_img = self.img_crop(self.image_data, xmin, ymin, xmax, ymax)
            img_water_idx= np.where(crop_img>235)
            img_gray_idx= np.where(crop_img<180)
            crop_img[img_water_idx]=255
            crop_img[img_gray_idx]=0
            bbox_ocr = bbox
            bbox_ocr['text'] = (self.rec_warper(crop_img)).upper()
            ocr_list.append(bbox_ocr)
        rec_end = time.time()
        print("Total time : {}, det time : {}, rec time : {}".format(rec_end - det_start, det_end - det_start, rec_end - rec_start))
        return json.dumps({'textline_num' : len(ocr_list), 'textline' : ocr_list})
"""
        


if __name__ == '__main__':
    test_lst = sys.argv[1]
    test_out = sys.argv[2]
    fout = open(test_out, 'w')
    imgs = []
    imgs_lst = []
    textline_ocr = TextLineOCR()
    with open(test_lst, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            img = cv2.imread(line)
            textline_res = textline_ocr.rec([img])
            for res in textline_res:
                res = [unicode(r, 'utf-8').encode('utf-8') for r in res]
                print("{} {}".format(line, ''.join(res)))
                fout.write("{} {}\n".format(line, ''.join(res)))
    fout.close()

