import implementations as imp
import pandas as pd

images = imp.read_images(
    "/Users/sameergururajmathad/Documents/CSC - 481/Final Project/images", "gray"
)

images.resize(300)

imp.to_hog(images)
