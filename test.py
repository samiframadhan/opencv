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
    mask_yellow = cv2.inRange(hsv_img, (15,60,20), (25, 255, 255))
    color_mask = cv2.bitwise_or(mask_white, mask_yellow)
    masked_img = np.copy(img)
    masked_img[mask_yellow == 0] = [0,0,0]

    gray_image = grayscale(masked_img)
    # plt.imshow(gray_image)
    # plt.show()

    kernel_size = 5
    blurred_gray_img = gaussian_blur(gray_image, kernel_size=kernel_size)
    # plt.imshow(blurred_gray_img)
    # plt.show()

    low_thresh = 50
    high_thresh = 90
    edges_img = canny(blurred_gray_img, low_threshold=low_thresh, high_threshold=high_thresh)
    # plt.imshow(edges_img)
    # plt.show()

    imshape = img.shape
    vertices = np.array(
        [
            [
                (0,imshape[0]),
                (3*imshape[1]/9, 5.5*imshape[0]/10), 
                (6*imshape[1]/9, 5.5*imshape[0]/10), 
                (imshape[1],imshape[0])
            ]
        ], 
        dtype=np.int32)
    
    polygon_img = np.copy(img)
    cv2.polylines(polygon_img, vertices, isClosed=True, color=(255,0,0), thickness=3)
    # plt.imshow(polygon_img)
    # plt.title("Polygon of Region of Interest")
    plt.show()
    
    masked_edges = region_of_interest(edges_img, vertices=vertices)
    # plt.imshow(masked_edges)
    # plt.show()

    rho = 2
    theta = np.pi / 180
    threshold = 15
    min_line_len = 8
    max_line_gap = 5
    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    # plt.imshow(line_img)
    # plt.show()

    overlay_img = weighted_img(line_img, img)
    # plt.imshow(overlay_img)
    # plt.show()

    return overlay_img

def process_image(img):
    result = lane_finding_pipeline(img=img)
    return result

# img = mpimg.imread('data/frame_0003.jpg')

output = 'output_v3.mp4'
clip = VideoFileClip('VideoTrack3.mp4')
out_clip = clip.fl_image(process_image)
out_clip.write_videofile(output, audio=False)

# plt.imshow(img)
# plt.show()

# result = lane_finding_pipeline(img)

# plt.imshow(result)
# plt.show()

# cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
# cap.set(cv2.CAP_PROP_EXPOSURE, 200)
# while cap.isOpened():
#     ret, frame = cap.read()
#     result = lane_finding_pipeline(ret)
#     cv2.imshow('frame', result)
#     k = cv2.waitKey(1) & 0xFF
#     if k == ord('s'):
#         cv2.imwrite('image.png', frame)
#     elif k == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()