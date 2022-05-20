import numpy as np
import cv2 as cv
import os
video_file = "/home/ubuntu/data/mastoid_video/V001/V001_part1.mp4"
output_dir = "/home/ubuntu/data/optical_flow_test"
output_dir = "/home/ubuntu/data/img_diff_test"

def align_two_imgs(img1_color, img2_color):
    img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY)
    height, width = img2.shape
    
    # Create ORB detector with 5000 features.
    orb_detector = cv.ORB_create(500)
    
    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)
    
    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
    
    # Match the two sets of descriptors.
    matches = list(matcher.match(d1, d2))

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key = lambda x: x.distance)
    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*0.9)]
    no_of_matches = len(matches)
    
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    
    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt
    
    # Find the homography matrix.
    homography, mask = cv.findHomography(p1, p2, cv.RANSAC)
    
    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv.warpPerspective(img1_color,
                        homography, (width, height))
    
    return transformed_img


cap = cv.VideoCapture(video_file)
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

index = 0
length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
while index < length:
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    print(f'frame{index}')
    # next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    # flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # hsv[..., 0] = ang*180/np.pi/2
    # hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    # bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    of_file = os.path.join(output_dir, f'frame{index}.png')
    # cv.imwrite(of_file, bgr)
    # prvs = next

    transformed_img1 = align_two_imgs(frame2, frame1)
    diff = transformed_img1 - frame1

    cv.imwrite(of_file, transformed_img1)
    prvs = frame2

    index += 10
    cap.set(cv.CAP_PROP_POS_FRAMES, index)

    

