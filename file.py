import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
images=np.zeros((4,2),np.int_)
framewidth = 320
frameheight = 240
counter=0
'''
canny dectection 
what is it ?
it is used to detect edeges in an image , 

'''
def callback(input):
     pass
def rearrange_points(pts):
    rect=np.zeros((4,2),dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]
    '''
        top left = has the smallest x+y ,(0,0)
        bottom right has the largest x+y
    '''
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]
    return rect

 
cap = cv.VideoCapture(0)
cap.set(3, framewidth)
cap.set(4, frameheight)
winname='canny'
cv.namedWindow(winname)

#191,255


captured_cards = []

while True:
    success, img = cap.read() 
    if not success:
        break  # If the frame was not successfully read, exit the loop

    imgcon = img.copy()
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    median = np.median(gray_img)
    sigma = 0.33  # A constant factor
    lower_threshold = int(max(0, (1.0 - sigma) * median))
    upper_threshold = int(min(255, (1.0 + sigma) * median))
    cannyedge=cv.Canny(gray_img,lower_threshold,upper_threshold)
    '''
    This detects the edges of the image 
    when we run this we gte a binary image 
    white Pixels(255) = edges 
    black pixels (0) = background 
    there are three parameters in this 
    gray_img -> we take the gray scale version of our image (or video) as edges depend on brightness not colour 
    we then have lower_threshold and upper_threshold and we compare it to the gradient 
    Gradient refers to rate of change 
    In images this refers to how quickly brightness changes from one pixel to another 
    if pixels are similar -> gradients are small ( flat area )
    if pixels change sharply -> gradients are large 
    so canny computes how quickly the brightness chnages 

    if Gradient > upper_threshold:
        its a strong edge so we keep it 
    else if upper_threshold > Gradient > lower_threshold:
        used to fill in the gabs so if this weaker edge is close to 
        a strong edge we keep it 
    else if lower_threshold > Gradient:
        Not a edge 
    '''
    contours, _ = cv.findContours(cannyedge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    '''
    takes 3 inputs 
    cannyedge -> takes the edges (points) from the cannyedge
    we use cv.findContour which works like a connect the dots tool. 
    It takes the white edge pixels from canny edge and traces them into continuous boundaries called contours.
    cv.RETR_EXTERNAL -> Outer boundary detection 
    cv.CHAIN_APPROX_SIMPLE -> OpenCV stores only the essential points needed to define the contour  the corners for polygons.
    if we used cv.CHAIN_APPROX_NONE then every single edge pixel is stored in the contour.
    '''

    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:2] 
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if area < 5000:  # ignore small contours
            continue
        hull = cv.convexHull(contour)
        solidity = area / cv.contourArea(hull)
        if solidity < 0.9:  # likely not a rectangle (could be circular like a chip)
            continue

        perimeter = cv.arcLength(contour,True)
        eplison = 0.02*perimeter
        approx = cv.approxPolyDP(contour,eplison,True)
        '''
            contour = The contour you got from the cv.findContours (list of boundary lines)

            True: make sure the polygon is closed connection the last point to the first point
            polygon = A shape formed by connecting a sequence of points with straight edges  where the last point connects back to the first point forming a loop
            Example 
            let say your contour is simplifed to 4 points 
            [(10, 10), (100, 10), (100, 200), (10, 200)]

            With the last parameter set as a True it makes it a closed loop
            10,10 -> 100,10 -> 100,200 -> 10,200 -> back to 10,10
        '''
        if len(approx) == 4:
            
            #Ensure the contour approximates to a polygon with 4 corners (likely a rectangular shape)
            print("Got 4 corners:", approx)
            pts = np.float32(approx.reshape(4,2))

            cv.imshow('canny',cannyedge)
            width, height = 400, 600
            ordered_pts = rearrange_points(pts)
            p1 = np.float32(images)
            p2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            matrix = cv.getPerspectiveTransform(ordered_pts, p2)
            output = cv.warpPerspective(img, matrix, (width, height))
            if len(approx) == 4 and area > 5000:
                captured_cards.append(output)
                cv.imshow(f"Card {i}", output)
                cv.drawContours(imgcon, [approx], -1, (255, 0, 255), 4)
                x, y, w, h = cv.boundingRect(approx)
                #Gte the x,y,w,h of the approx 
                cv.rectangle(imgcon, (x, y), (x + w, y + h), (0, 255, 0), 3)
                #Place text on image 
                cv.putText(imgcon, "Card", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv.LINE_AA)
                cv.imshow("Detected Cards", imgcon)
                
    cv.namedWindow("Card")
    cv.imshow('Card', gray_img)
    key = cv.waitKey(30) & 0xFF
    if key == ord('c') and captured_cards:
        for idx, card in enumerate(captured_cards):
            cv.imshow(f"Captured Card {idx}", card)
            zoom_width = 100
            zoom_height = 200
            cropped_card = captured_cards[idx]
            top_left_region = cropped_card[0:zoom_height, 0:zoom_width]
            zoomed_display = cv.resize(top_left_region, (200, 300))  
            cv.imshow(f"Top-Left Zoom {idx}", zoomed_display)
            imghsv=cv.cvtColor(zoomed_display,cv.COLOR_BGR2HSV)

            lower = np.array([0, 0, 200])   
            upper = np.array([179, 100, 255]) 
            
            mask=cv.inRange(imghsv,lower,upper)
            cv.imshow(f"White Mask {idx}", mask)
            contour_mask, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        print("Captured still image!")

    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
