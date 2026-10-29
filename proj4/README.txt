cs180 project 4a: image warping and mosaicing

what is here
homography.py: computeH, the least-squares system, warpImageNearestNeighbor and warpImageBilinear (inverse warping, own interpolation).
mosaic.py: warping a set of images into one frame, distance-transform weights, overwrite, weighted average and two-band blending, and the cylindrical projection.
pick_points.py: correspondences. raw points live in points/raw, the snapped points it writes live in points. with a third argument it opens a matplotlib window to click points.
images.py: loads a photo from data at quarter size and saves results.
main.py: makes every figure and media/results.json.
build_page.py: writes index.html from media.
points/rectify.json: the four corners and target size for each rectification.

how to run
put the photos in data as IMG_xxxx.jpg (they are gitignored). i converted the iphone heic files with heif-convert so the rotation is applied.

    ../.venv/bin/python pick_points.py IMG_4048 IMG_4049
    ../.venv/bin/python main.py
    ../.venv/bin/python build_page.py

pick_points.py IMG_4048 IMG_4049 re-snaps the raw points for that pair. pick_points.py IMG_4048 IMG_4049 8 asks for 8 new clicks in each image first.
main.py takes about 30 seconds.
