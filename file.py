import cv2 as cv
import numpy as np

framewidth = 320
frameheight = 240

def rearrange_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left which is the smallest x and y
    rect[3] = pts[np.argmax(s)]   # bottom-right which is the largest x and y

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top right
    rect[2] = pts[np.argmax(diff)]  # bottom left
    return rect

def process_card(card, idx=0):
    """
    Takes a rectified card image and:
    - crops the top-left corner
    - zooms it
    - converts to HSV
    - creates a white mask
    - shows the zoom + mask
    """
    zoom_width = 100
    zoom_height = 200
    top_left_region = card[0:zoom_height, 0:zoom_width]

    zoomed_display = cv.resize(top_left_region, (200, 300))
    cv.imshow(f"Zoom {idx+1}", zoomed_display)

    imghsv = cv.cvtColor(zoomed_display, cv.COLOR_BGR2HSV)

    lower = np.array([0, 0, 200])
    upper = np.array([179, 100, 255])
    mask = cv.inRange(imghsv, lower, upper)
    cv.imshow(f"Mask {idx+1}", mask)

    contour_mask, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print(f"Card {idx}: {len(contour_mask)} white regions found")

    return zoomed_display, mask, contour_mask


# setting up camera here
cap = cv.VideoCapture(0)
cap.set(3, framewidth)
cap.set(4, frameheight)

cv.namedWindow("Live")
cv.namedWindow("Canny")

captured_count = 0           # how many cards captured so far (we need 2)
card_present_prev = False    # checking if there was there a card in the previous frame

while True:
    success, img = cap.read()
    if not success:
        print("Camera frame not read, exiting.")
        break

    imgcon = img.copy()
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Show the live grayscale feed
    cv.imshow("Live", gray_img)

    # Auto Canny thresholds based on median
    median = np.median(gray_img)
    sigma = 0.33
    lower_threshold = int(max(0, (1.0 - sigma) * median))
    upper_threshold = int(min(255, (1.0 + sigma) * median))

    cannyedge = cv.Canny(gray_img, lower_threshold, upper_threshold)
    cv.imshow("Canny", cannyedge)

    # Find contours
    contours, _ = cv.findContours(cannyedge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:2]

    # Collect valid card-like quads in this frame
    card_quads = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area < 5000:
            continue

        hull = cv.convexHull(contour)
        hull_area = cv.contourArea(hull)
        if hull_area == 0:
            continue

        solidity = area / hull_area
        if solidity < 0.9:
            continue

        perimeter = cv.arcLength(contour, True)
        epsilon = 0.02 * perimeter
        approx = cv.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            card_quads.append(approx)

    card_present_now = len(card_quads) > 0

    # Edge trigger: cards just appeared this frame
    if (not card_present_prev) and card_present_now and captured_count < 2:
        print(f"New card event: {len(card_quads)} card(s) in this frame")

        # How many slots left until we hit 2
        slots_left = 2 - captured_count

        for i, approx in enumerate(card_quads[:slots_left]):
            idx = captured_count + i
            print(f"Capturing card index {idx}")

            pts = np.float32(approx.reshape(4, 2))
            width, height = 400, 600
            ordered_pts = rearrange_points(pts)
            p2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            matrix = cv.getPerspectiveTransform(ordered_pts, p2)
            output = cv.warpPerspective(img, matrix, (width, height))

            # Show rectified card
            cv.imshow(f"Captured Card {idx+1}", output)

            # Draw on original (optional)
            cv.drawContours(imgcon, [approx], -1, (255, 0, 255), 4)
            x, y, w, h = cv.boundingRect(approx)
            cv.rectangle(imgcon, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv.putText(imgcon, f"Card {idx+1}", (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            cv.imshow("Detected Cards", imgcon)

            # Process inner corner
            process_card(output, idx=idx)

        captured_count += min(len(card_quads), slots_left)
        print(f"Total captured so far: {captured_count}")

        if captured_count >= 2:
            print("2 cards captured — stopping camera. Press any key to close.")
            break

    # Update state for next frame
    card_present_prev = card_present_now

    # While still scanning you can quit with q
    key = cv.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cap.release()
# Keep all result windows on screen until a key is pressed
cv.waitKey(0)
cv.destroyAllWindows()
