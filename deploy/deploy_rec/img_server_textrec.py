# -*- coding:utf-8 -*-
import json
import base64
import falcon
from img_log import api
import cv2
import numpy as np

from ocr import OCR
text_rec=OCR()

class Validate(object):
    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.body = 'Service started!'


class ForwardImg(object):
    def on_post(self, req, resp):
	res = None
        img = ''
        # Data comes from business.
        while True:
            chunk = req.stream.read(4096)
            if not chunk:
                break
            img += chunk
        # img = req.stream.read()
        # content length 校验
        content_length = int(req.headers.get('CONTENT-LENGTH', 0))
        if content_length == 0:
            resp.status = falcon.HTTP_411
            resp.body = json.dumps({'status': 411,
                                    'msg': 'Content length needed.'})
            return
        if len(img) != content_length:
            resp.status = falcon.HTTP_400
            resp.body = json.dumps({'status': 400,
                                    'msg': 'Image not complete.'})
            return

        try:
            img_data = json.loads(img)
            img = base64.b64decode(img_data['img'])
            #save_img = open('imgin.png','wb')
            #save_img.write(img)
            #save_img.close()
            det_res = img_data['det_res']
            dets = det_res['bboxes']
            #api.logger.info('img_data : {}'.format(img_data))
            #api.logger.info('Img : {}'.format(img))
            api.logger.info('dets : {}'.format(dets))
            res = text_rec.rec_warper_with_dets(img, dets)
            
            #res = text_rec.rec_warper(img)
            resp.body = res
        except Exception as e:
	    api.logger.error('[-] server error {}.'.format(e))
            resp.status = falcon.HTTP_500
            json.dumps({'status': 500,
                         'msg': str(e)})
            return

        if res is None :
	    api.logger.error('[-] res: ' + str(res))
            resp.status = falcon.HTTP_500
            resp.body = json.dumps({'status': 500,
                                    'msg': 'Image analyzer returned None.'})
        else:
	    api.logger.info('[+] res: ' + str(res))
            resp.body = res 


app = falcon.API()
forward_img = ForwardImg()
validate_service = Validate()

app.add_route('/forward', forward_img)
app.add_route('/test', validate_service)

if __name__ == '__main__':
    from wsgiref import simple_server

    httpd = simple_server.make_server('0.0.0.0', 4067, app)
    httpd.serve_forever()
