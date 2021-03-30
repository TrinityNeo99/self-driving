# encoding:utf-8
import requests
import json
import base64
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# client_id 为官网获取的AK， client_secret 为官网获取的SK
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=58hrg6MpikP12p39PrbQuGlw&client_secret=f7mGLpUUyxAs57zEIc03xMOkm290lxRz'
response = requests.get(host)
if response:
    print(response.json())
    dct = json.dumps(response.json())
    print(dct)

'''
图像主体检测
'''

request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/object_detect"
# 二进制方式打开图片文件
image_name = "/Users/neotrinity/Downloads/Papper.jpeg"
# image_name = "/Users/neotrinity/Downloads/traffic.jpg"
# image_name = "/Users/neotrinity/Downloads/traffic2.jpeg"
image_name = '/Users/neotrinity/Downloads/traffic3.jpeg'
image_name = '/Users/neotrinity/Downloads/traffic4.jpeg'
image_name = '/Users/neotrinity/Downloads/traffic5.jpeg'
f = open(image_name, 'rb')
img = base64.b64encode(f.read())

params = {"image":img,"with_face":0}
access_token = '24.756d8f404937ef73dd6312d6ea046a40.2592000.1612597490.282335-23502003'
request_url = request_url + "?access_token=" + access_token
headers = {'content-type': 'application/x-www-form-urlencoded'}
response = requests.post(request_url, data=params, headers=headers)
if response:
    result = response.json()

print(result)
left = result['result']["left"]
top = result['result']["top"]
width = result['result']["width"]
height = result['result']["height"]

import cv2
image = cv2.imread(image_name)
cv2.rectangle(image, (left, top), (left + width, top + height), (255, 0, 0), 2)
cv2.imwrite('2.jpg', image)
im = plt.imread("2.jpg")
plt.imshow(im)
plt.show()


