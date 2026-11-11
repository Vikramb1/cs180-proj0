cs180 project 4: [auto]stitching photo mosaics (parts a and b)

part a, by hand
homography.py: computeH, the least-squares system, warpImageNearestNeighbor and warpImageBilinear (inverse warping, own interpolation).
mosaic.py: warping a set of images into one frame, distance-transform weights, overwrite, weighted average and two-band blending, and the cylindrical projection.
pick_points.py: correspondences. raw points live in points/raw, the snapped points it writes live in points. with a third argument it opens a matplotlib window to click points.
images.py: loads a photo from data at quarter size and saves results.
main.py: makes every part a figure and media/results.json.
points/rectify.json: the four corners and target size for each rectification.

part b, automatic
harris.py: the course's harris detector (get_harris_corners) and dist2, with the comments and docstrings removed.
features.py: ANMS, descriptors (axis-aligned or rotated to a corner direction), the ratio-test matcher and 4-point RANSAC.
auto.py: makes every part b figure and media/results_b.json (corners, descriptors, matches, ransac, automatic mosaics, rotation test).
compare.py: scores manual against automatic homographies on the overlap and draws the ratio histogram.

web page
build_page.py: writes index.html from media (both parts).

how to run
put the photos in data as IMG_xxxx.jpg (they are gitignored). i converted the iphone heic files with heif-convert so the rotation is applied.

    ../.venv/bin/python pick_points.py IMG_4048 IMG_4049
    ../.venv/bin/python main.py
    ../.venv/bin/python auto.py
    ../.venv/bin/python compare.py
    ../.venv/bin/python build_page.py

pick_points.py IMG_4048 IMG_4049 re-snaps the raw points for that pair. pick_points.py IMG_4048 IMG_4049 8 asks for 8 new clicks in each image first.
main.py takes about 30 seconds. auto.py takes about 2 minutes, most of it in the harris detector.
