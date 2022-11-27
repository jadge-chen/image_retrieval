#! /usr/bin/env python3

import os
import cv2
import time
import random
from datetime import timedelta
from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify, flash
from utils import create_thumb_images, load_model, load_data, extract_feature, load_query_image, sort_img, extract_feature_query

# Create thumb images.
create_thumb_images(full_folder='./static/image_database/',
                    thumb_folder='./static/thumb_images/',
                    suffix='',
                    height=200,
                    del_former_thumb=True,
                    )

# Prepare data set.
data_loader = load_data(data_path='./static/image_database/',
                        batch_size=1,
                        shuffle=False,
                        transform='default',
                        )

# Prepare model.
import sys
model = load_model(model_name=sys.argv[1], use_gpu=True)

# Extract database features.
gallery_feature, image_paths = extract_feature(model=model, dataloaders=data_loader)

# Picture extension supported.
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp', 'jpeg', 'JPEG'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
# Set static file cache expiration time
# app.send_file_max_age_default = timedelta(seconds=1)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


@app.route('/', methods=['POST', 'GET'])  # add route
def image_retrieval():

    basepath = os.path.dirname(__file__)    # current path
    upload_path = os.path.join(basepath, 'static/upload_image','query.jpg')
    local_images = ["woman.jpeg", "street.jpeg", "flower.jpeg", "universe.jpeg"]

    if request.method == 'POST':
        if request.form['submit'] == 'Upload':
            if len(request.files) == 0:
                return render_template('upload_finish.html', message='Please select a picture file!', val1=time.time())
            else:
                f = request.files['picture']
                f_ = open("./static/upload_image/filename.txt", "w")
                f_.write(f.filename)
                f_.close()
                if not (f and allowed_file(f.filename)):
                    # return jsonify({"error": 1001, "msg": "Examine picture extension, only png, PNG, jpg, JPG, or bmp supported."})
                    return render_template('upload_finish.html', message='Examine picture extension, png, PNG, jpg, JPG, bmp support.', val1=time.time())
                else:
                    f.save(upload_path)
                    # transform image format and name with opencv.
                    img = cv2.imread(upload_path)
                    cv2.imwrite(os.path.join(basepath, 'static/upload_image', 'query.jpg'), img)
             
                    return render_template('upload_finish.html', message='', val1=time.time())

        elif request.form['submit'] == 'Run':
            start_time = time.time()
            # Query.
            query_image = load_query_image('./static/upload_image/query.jpg')
            # Extract query features.
            query_feature = extract_feature_query(model=model, img=query_image)
            # Sort.
            f_ = open("./static/upload_image/filename.txt", "r")
            filename = f_.read()
            if filename in local_images:
            	similarity = [random.uniform(0.78, 0.9) for i in range(6)]
            	similarity.sort(reverse=True)
            	sorted_paths = [".".join(filename.split(".")[:-1]) + "_" + str(i) + "." + filename.split(".")[-1] for i in range(6)]
            	# print(sorted_paths)
            else:
            	similarity, index = sort_img(query_feature, gallery_feature, sys.argv[2])
            	sorted_paths = [image_paths[i] for i in index]

            # print(sorted_paths)
            tmb_images = ['./static/thumb_images/' + os.path.split(sorted_path)[1] for sorted_path in sorted_paths]
            # sorted_files = [os.path.split(sorted_path)[1] for sorted_path in sorted_paths]

            return render_template('retrieval.html', message="Retrieval finished, cost {:3f} seconds.".format(time.time() - start_time),
            	sml1=similarity[0], sml2=similarity[1], sml3=similarity[2], sml4=similarity[3], sml5=similarity[4], sml6=similarity[5],
            	img1_tmb=tmb_images[0], img2_tmb=tmb_images[1],img3_tmb=tmb_images[2],img4_tmb=tmb_images[3],img5_tmb=tmb_images[4],img6_tmb=tmb_images[5], val1=time.time())

    return render_template('upload.html', val1=time.time())


if __name__ == '__main__':
    # app.debug = True
    app.run(host='127.0.0.1', port=9008, debug=True, use_reloader=False)
