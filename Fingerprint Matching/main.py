import os

import cv2 as cv

REAL_IMAGES_DIR = "Fingerprints/Real/"
TEST_IMAGE_PATH = "Fingerprints/Altered/1__M_Right_index_finger_CR.BMP"

altered_image = cv.imread(TEST_IMAGE_PATH)

max_score = 0
filename = None
real_image = None
kp1 = kp2 = gm = None

for file in os.listdir(REAL_IMAGES_DIR):
    fingerprint_image = cv.imread(REAL_IMAGES_DIR + file)

    sift = cv.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(altered_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(fingerprint_image, None)

    keypoints = len(min([keypoints1, keypoints2], key=len))

    index_params = dict(algorithm=1, trees=10)  # Use KDTree, with 10 trees
    search_params = dict(checks=50)  # Number of times the trees should be checked
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.1 * n.distance]

    score = len(good_matches) / keypoints * 100

    if score > max_score:
        max_score = score
        filename = file
        real_image = fingerprint_image
        kp1, kp2, gm = keypoints1, keypoints2, good_matches


print("BEST MATCH: " + filename)
print("\nSCORE: " + str(max_score))

matches = cv.drawMatches(altered_image, kp1, real_image, kp2, gm, None)
matches = cv.resize(matches, None, fx=5, fy=5)
cv.imshow("Matches (Real image on the right)", matches)
cv.waitKey(0)
cv.destroyAllWindows()
