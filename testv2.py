import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from moviepy.editor import VideoFileClip

# video_backend = [cv2.videoio_registry.getBackendName(i) for i in cv2.videoio_registry.getBackends()]
# print(video_backend)
# print(cv2.CAP_FFMPEG)
# print(cv2.CAP_GSTREAMER)
# print(cv2.CAP_INTEL_MFX)
# print(cv2.CAP_V4L2)
# print(cv2.CAP_IMAGES)

## User-defined parameters: (Update these values to your liking)
# Minimum size for a contour to be considered anything
MIN_AREA = 500 

# Minimum size for a contour to be considered part of the track
MIN_AREA_TRACK = 5000

# Robot's speed when following the line
LINEAR_SPEED = 0.2

# Proportional constant to be applied on speed when turning 
# (Multiplied by the error value)
KP = 1.5/100 

# If the line is completely lost, the error value shall be compensated by:
LOSS_FACTOR = 1.2

# Send messages every $TIMER_PERIOD seconds
TIMER_PERIOD = 0.06

# When about to end the track, move for ~$FINALIZATION_PERIOD more seconds
FINALIZATION_PERIOD = 4

# The maximum error value for which the robot is still in a straight line
MAX_ERROR = 30

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def crop_size(height, width):
    """
    Get the measures to crop the image
    Output:
    (Height_upper_boundary, Height_lower_boundary,
     Width_left_boundary, Width_right_boundary)
    """
    ## Update these values to your liking.

    return (1*height//3, height, width//4, 3*width//4)

def get_contour_data(mask):
    """
    Return the centroid of the largest contour in
    the binary image 'mask' (the line) 
    and return the side in which the smaller contour is (the track mark) 
    (If there are any of these contours),
    and draw all contours on 'out' image
    """ 
    # get a list of contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mark = {}
    line = {}

    crop_w_start = 0
    # print(contours)

    for contour in contours:
        
        M = cv2.moments(contour)
        # Search more about Image Moments on Wikipedia :)

        if M['m00'] > MIN_AREA:
        # if countor.area > MIN_AREA:

            if (M['m00'] > MIN_AREA_TRACK):
                # Contour is part of the track
                line['x'] = int(M["m10"]/M["m00"])
                line['y'] = int(M["m01"]/M["m00"])

                # plot the area in light blue
                # cv2.drawContours(out, contour, -1, (255,255,0), 1) 
                # cv2.putText(out, str(M['m00']), (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])),
                # cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
            
            else:
                # Contour is a track mark
                if (not mark) or (mark['y'] > int(M["m01"]/M["m00"])):
                    # if there are more than one mark, consider only 
                    # the one closest to the robot 
                    mark['y'] = int(M["m01"]/M["m00"])
                    mark['x'] = int(M["m10"]/M["m00"])

                    # plot the area in pink
                    # cv2.drawContours(out, contour, -1, (255,0,255), 1) 
                    # cv2.putText(out, str(M['m00']), (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])),
                    # cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)


    if mark and line:
    # if both contours exist
        if mark['x'] > line['x']:
            mark_side = "right"
        else:
            mark_side = "left"
    else:
        mark_side = None


    return (line, mark_side, contours)

def draw_contours(mask, contours):
    for contour in contours:
        cv2.drawContours(mask, contour, -1, (255,0,255), 1)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

previous_lines = [0, 0, 0, 0]
    
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    Draw every line segment from `lines` and then extrapolate and draw 
    the full lane lines (left and right) using averaging and smoothing.
    """
    # Draw every line segment first
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    # # Get image shape (used for extrapolation)
    # imshape = img.shape
    # # Initialize lists to hold line segments
    # left_lines = []
    # right_lines = []

    # # Separate line segments based on slope
    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         # Calculate slope (m) and intercept (b) for the line
    #         m = (y2 - y1) / (x2 - x1)
    #         b = y1 - m * x1
    #         if m < 0:
    #             left_lines.append((m, b))
    #         elif m > 0:
    #             right_lines.append((m, b))

    # # Process left line
    # if len(left_lines) > 0:
    #     left_m = [line[0] for line in left_lines]
    #     left_b = [line[1] for line in left_lines]
    #     ml = np.mean(left_m)
    #     bl = np.mean(left_b)

    #     # Extrapolate left line
    #     y1l = imshape[0]  # bottom of the image
    #     y2l = int(6 * imshape[0] / 10)  # middle of the image
    #     x1l = int((y1l - bl) / ml)
    #     x2l = int((y2l - bl) / ml)

    #     cv2.line(img, (x1l, y1l), (x2l, y2l), [0, 255, 0], thickness)  # draw the left line

    # # Process right line
    # if len(right_lines) > 0:
    #     right_m = [line[0] for line in right_lines]
    #     right_b = [line[1] for line in right_lines]
    #     mr = np.mean(right_m)
    #     br = np.mean(right_b)

    #     # Extrapolate right line
    #     y1r = imshape[0]  # bottom of the image
    #     y2r = int(6 * imshape[0] / 10)  # middle of the image
    #     x1r = int((y1r - br) / mr)
    #     x2r = int((y2r - br) / mr)

    #     cv2.line(img, (x1r, y1r), (x2r, y2r), [0, 0, 255], thickness)  # draw the right line

    # Store the current line for smoothing in future frames (if needed)
    # previous_lines[0] = ml if len(left_lines) > 0 else previous_lines[0]
    # previous_lines[1] = bl if len(left_lines) > 0 else previous_lines[1]
    # previous_lines[2] = mr if len(right_lines) > 0 else previous_lines[2]
    # previous_lines[3] = br if len(right_lines) > 0 else previous_lines[3]               
    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # print(lines)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=0.6, γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def lane_finding_pipeline(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask_white = cv2.inRange(img, (200,200,200), (255, 255, 255))
    mask_yellow = cv2.inRange(hsv_img, (40, 60, 30), (100, 255, 255))
    color_mask = cv2.bitwise_or(mask_white, mask_yellow)
    masked_img = np.copy(img)
    masked_img[mask_yellow == 0] = [0,0,0]

    gray_image = grayscale(masked_img)
    # cv2.show()

    kernel_size = 5
    blurred_gray_img = gaussian_blur(gray_image, kernel_size=kernel_size)
    # cv2.waitKey(5)
    # cv2.imshow("gray", blurred_gray_img)
    # plt.imshow(blurred_gray_img)
    # plt.show()

    low_thresh = 50
    high_thresh = 90
    edges_img = canny(blurred_gray_img, low_threshold=low_thresh, high_threshold=high_thresh)
    # plt.imshow(edges_img)
    # plt.show()

    imshape = img.shape

    # Trapesium shape
    #     x1,y1         x2,y2
    #       --------------
    #     /                \
    #    --------------------
    # x0,y0                 x3,y3
    vertices = np.array(
        [
            [
                (0,imshape[0]), #x0, y0
                (3*imshape[1]/9, 5*imshape[0]/10), #x1,y1
                (6*imshape[1]/9, 5*imshape[0]/10), #x2,y2
                (imshape[1],imshape[0]) #x3,y3
            ]
        ], 
        dtype=np.int32)
    
    polygon_img = np.copy(img)
    cv2.polylines(polygon_img, vertices, isClosed=True, color=(255,0,0), thickness=3)
    # cv2.imshow("poly",polygon_img)
    # cv2.waitKey(1)
    # plt.imshow(polygon_img)
    # plt.title("Polygon of Region of Interest")
    # plt.show()
    
    masked_edges = region_of_interest(edges_img, vertices=vertices)
    maskv2 = region_of_interest(blurred_gray_img, vertices)
    # plt.imshow(masked_edges)
    # plt.show()

    rho = 2
    theta = np.pi / 180
    threshold = 15
    min_line_len = 8
    max_line_gap = 5
    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    # centroid = 
    # plt.imshow(line_img)
    # plt.show()

    cv2.imshow("mask", masked_img)
    cv2.waitKey(5)
    line, mark_side, contours = get_contour_data(maskv2)
    
    # print(line)
    if line:
        cv2.circle(line_img, (line['x'], line['y']), 5, (0,255,0), 7)

    overlay_img = weighted_img(line_img, img)
    # plt.imshow(overlay_img)
    # plt.show()

    return (overlay_img, maskv2)

def create_histogram(mask):
    hsv_img = mask

    # Define channel names and colors for plotting
    channels = ('Hue', 'Saturation', 'Value')
    colors = ('r', 'g', 'b')

    plt.figure()
    plt.title("HSV Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    # Calculate and plot the histogram for each HSV channel
    for i, channel in enumerate(channels):
        hist = cv2.calcHist([hsv_img], [i], None, [256], [0, 256])
        plt.plot(hist, color=colors[i], label=channel)
        plt.xlim([0, 256])

    plt.legend()
    plt.show()

def process_image(img):
    result = lane_finding_pipeline(img=img)
    return result

# img = mpimg.imread('data/frame_0003.jpg')

# output = 'output_v2.mp4'
# clip = VideoFileClip('VideoTrack.mp4')
# out_clip = clip.fl_image(process_image)
# out_clip.write_videofile(output, audio=False)

# plt.imshow(img)
# plt.show()

# result = lane_finding_pipeline(img)

# plt.imshow(result)
# plt.show()

cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
# cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('R', 'G', 'B', ' '))
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, 200)
loadmask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
while cap.isOpened():
    ret, frame = cap.read()
    result, mask = lane_finding_pipeline(frame)
    
    # result = cv2.bitwise_and(frame, frame, mask=loadmask)
    # create_histogram(frame)
    cv2.imshow('frame', result)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        cv2.imwrite('image.png', frame)
        cv2.imwrite('mask.png', mask)
    elif k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()