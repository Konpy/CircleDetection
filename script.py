import numpy as np
from PIL import ImageGrab
import cv2
import time


def roi(img,vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,vertices,255)
    masked = cv2.bitwise_and(img,mask)
    return masked

def draw_circles(img,circle):
    if circle is not None:
        circle = np.uint16(np.around(circle))
        for i in circle[0,:]:
            center = (i[0], i[1])
                # draw the outer circle
            radius = i[2]
            cv2.circle(img, center, radius, (255, 0, 255), 3)


def process_img(image):
    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection  
    processed_img =  cv2.Canny(processed_img, threshold1 = 250, threshold2=500)
    #blur the img
    processed_img = cv2.GaussianBlur(processed_img,(5,5),1)
    #x,y
    vertices = np.array([[0,55],[800,55],[800,570],[0,570]], np.int32)
    processed_img = roi(processed_img, [vertices])
    rows = processed_img.shape[0]
    circles = cv2.HoughCircles(processed_img,cv2.HOUGH_GRADIENT,1,rows/8,param1=50,param2=30,minRadius = 30,maxRadius=100)
    draw_circles(processed_img,circles)
    return processed_img

def main():
    last_time = time.time()
    while True:
        screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        new_screen = process_img(screen)  
        #cv2.imshow('window', new_screen)
        cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
main()
