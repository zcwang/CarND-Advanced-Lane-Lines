## Gary's Writeup for Advanced Lane Finding Project in CarND Self-Driving Cars program

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

please also refer "My Advanced Lane Lines.ipynb" and "My Advanced Lane Lines.pdf" for this README.

### Camera Calibration

#### 1. Compute the camera matrix and distortion coefficients.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

A. Camera caliberation with parameters

    # Call camera_cal_parameters() in lane_detection_util.py to get dist_matrix, dist_param
    ...
    # In camera_cal_parameters(), arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size,None,None)
    ...
    # undistort() in lane_detection_util.py
    undis_image = cv2.undistort(image, dist_matrix,dist_param, None, dist_matrix)
    ...

![alt text][image1]

### Pipeline (single images)

#### Do distortion-corrected image.
    ...
    image = images[4]
    # Call undistort() in lane_detection_util.py 
    image_undist = undistort(image, dist_matrix, dist_param)
    fig,(ax1,ax2) = plt.subplots(1, 2, figsize=(16,4))
    ax1.imshow(image)
    ax1.set_title('original', fontsize=10)
    ax2.imshow(image_undist)
    ax2.set_title('undistorted', fontsize=10)
    plt.show()
    image_undist = undistort(image, dist_matrix, dist_param)
    ...
Then, invoke cv2.undistor(),

    ...
    undis_image = cv2.undistort(image, dist_matrix,dist_param, None, dist_matrix)
    ...
    
![alt text][image2]

#### 2. Do image color transforms, gradients or other methods to create a thresholded binary image.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `lane_detection_util.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

    ...
    # Call preprocess() in lane_detection_util.py for color_threshold, gray_threshold processing
    image_binary = preprocess(image_undist, mask_vertices=mask_vertices, color_thresh=color_thresh, gray_thresh=gray_thresh)
    ...
    mask = np.uint8(np.zeros_like(image[:,:,0]))
    vertices = mask_vertices
    cv2.fillPoly(mask, vertices, (1))
    c_channel = np.max(image,axis=2)-np.min(image,axis=2)
    _,c_binary = cv2.threshold(c_channel,color_thresh[0],color_thresh[1],cv2.THRESH_BINARY)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    _, gray_binary = cv2.threshold(gray,gray_thresh[0],gray_thresh[1],cv2.THRESH_BINARY)
    combined_binary_masked = cv2.bitwise_and(cv2.bitwise_or(c_binary,gray_binary),mask)
    ...

![alt text][image3]

#### 3. Performe perspective transform.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

    ...
    # Call perspective_transform() in lane_detection_util.py for color_threshold, gray_threshold processing
    image_binary_bird = perspective_transform(image_binary, src, dst)
    ...
    img_size = img_binary.shape[::-1]
    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(img_binary, M, img_binary.shape[::-1], flags=cv2.INTER_LINEAR)
    ...

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Use gaussian smooth for image then find peaks of those data based on sliding window ( only the pixels in the sliding window were selected). Please refer select_lane_lines() in lane_detection_util.py

Then I did some fitting the left and right lane lines, return the quadratic coefficients and fit my lane lines with a 2nd order polynomial kinda like this:

    ...
    # Call select_lane_lines() in lane_detection_util.py
    left_lane, right_lane = select_lane_lines(image_binary_bird)
    ...
    # Call fit_lane_line() in lane_detection_util.py
    left_fit, right_fit = fit_lane_line([left_lane, right_lane])
    ...
    # fit the left and right lane pixels
    left_Y,left_X = np.where(left_lane_binary==1)
    left_fit = np.polyfit(left_Y, left_X, 2)
    left_fitx = left_fit[0]*left_Y**2 + left_fit[1]*left_Y + left_fit[2]

    right_Y,right_X = np.where(right_lane_binary==1)
    right_fit = np.polyfit(right_Y, right_X, 2)
    right_fitx = right_fit[0]*right_Y**2 + right_fit[1]*right_Y + right_fit[2]
    ...

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `lane_detection_util.py`, please refer cal_curvature().

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `lane_detection_util.py` in the function `draw_lane()`.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Use thresholding for histogram equalization, contrast/brightness adjustments or HDR should be right direction to do testing. But tuning parameters in OpenCV was not easy for me...QQ
