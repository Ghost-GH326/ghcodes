# from ultralytics import YOLO
from PIL import Image
import cv2, os, subprocess

# # Load a model
# model = YOLO('yolov8m-seg.pt')  # load an official model
# # model = YOLO('path/to/best.pt')  # load a custom model

# # # Predict with the model
# # image_path='/Applications/Recaptured_image_detection/91d7a751b12e4ad6a44816b86f3c271a_5rmYQUExNzg%3D_1709025128095.mp4-photo.jpg'
# folder_path='/Applications/Recaptured_image_detection/Final_integrity_test_dataset/Testing_dataset_entire_False'
# file_path='/Applications/Recaptured_image_detection/Final_integrity_test_dataset/Testing_dataset_entire_False'
# results = model(file_path)  # predict on an image
# results = model(file_path)  # predict on an image

# # Visualize the results
# for i, r in enumerate(results):
#     # Plot results image
#     im_bgr = r.plot()  # BGR-order numpy array
#     im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

#     # r.show()

#     # Save results to disk
#     r.save(filename=f'/Applications/Recaptured_image_detection/Final_integrity_test_dataset/Entire_False_result/results{i}.jpg')

def check_uniform(arr, compare_arr, max_diff=25):
    diff_count = (arr != compare_arr).sum(axis=(0, 1))
    return (diff_count <= max_diff).all()

def detect_uniform_regions(image_path, num_rows=2, num_columns=2):
    # Load the image
    image = cv2.imread(image_path)

    # Extract the top, bottom, left, and right regions
    height, width, _ = image.shape
    top_rows = image[:num_rows, :]
    bottom_rows = image[height - num_rows:, :]
    left_columns = image[:, :num_columns].reshape(num_columns,len(image),3)
    right_columns = image[:, width - num_columns:].reshape(num_columns, len(image),3)

    # Check if all pixels in the top rows have the same RGB values
    # top_uniform = (top_rows == top_rows[0]).all(axis=(0, 1)).all()
    # bottom_uniform = (bottom_rows == bottom_rows[0]).all(axis=(0, 1)).all()
    # left_uniform = (left_columns == left_columns[:, :1]).all(axis=(0, 1)).all()
    # right_uniform = (right_columns == right_columns[:, :1]).all(axis=(0, 1)).all()
    top_uniform = check_uniform(top_rows, top_rows[0])
    bottom_uniform = check_uniform(bottom_rows, bottom_rows[0])
    left_uniform = check_uniform(left_columns, left_columns[0])
    right_uniform = check_uniform(right_columns, right_columns[0])

    return top_uniform, bottom_uniform, left_uniform, right_uniform

def filt_images(image_path):
    command_1 = f'cp {image_path} /home/guohao826/fanpai_classification/data/screenshot'
    command_2 = f'cp {image_path} /home/guohao826/proper'
    command_3 = f'rm {image_path}'
    subprocess.run(command_1, shell=True)
    subprocess.run(command_2, shell=True)
    subprocess.run(command_3, shell=True)

# Example usage
image_folder = '/home/guohao826/fanpai_classification/data/fanpai_data_v2.0/except_images/test_uniform'
images = os.listdir(image_folder)
for i in images:
    image_path = os.path.join(image_folder, i)
    if 'jpg' not in i:
        subprocess.run(f'rm {image_path}', shell=True)
    top_uniform, bottom_uniform, left_uniform, right_uniform = detect_uniform_regions(image_path)

    if top_uniform or bottom_uniform or left_uniform or right_uniform:
        # filt_images(image_path)
        print(f'{i}:improper')


# image_folder = '/home/guohao826/proper'
# images = os.listdir(image_folder)
# for i in images:
#     image_path = os.path.join(image_folder, i)
#     if 'jpg' not in i:
#         imgcode = i.rsplit('.')[0]
#         subprocess.run(f'mv {image_path} {imgcode}.jpg', shell=True)

