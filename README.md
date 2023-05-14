# Image_Processing
Analyse the Image Meta-tags, then identifying the quality and understanding the purpose of the image. Classify the image and create keywords for it. Enhance low resolution images and discard poor resolution images.

In this the dags are being created using Apache Airflow, a scheduler to fetch data from queue one after the other and process the images and then push it to mongodb.

You can find the main class funtions from image_processing_metatag - dags - src - classes - class_image_utils.py -> class_metatag.py -> class_cleaning.py

The classes and scripts are being called in image_processing_metatag - dags - subscriber - subscriber.py


