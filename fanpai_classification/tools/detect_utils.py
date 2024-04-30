import cv2

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
    top_uniform = check_uniform(top_rows, top_rows[0])
    bottom_uniform = check_uniform(bottom_rows, bottom_rows[0])
    left_uniform = check_uniform(left_columns, left_columns[0])
    right_uniform = check_uniform(right_columns, right_columns[0])

    if top_uniform or bottom_uniform or left_uniform or right_uniform:
        return True
    return False