
import cv2
import numpy as np
from PIL import Image
from copy import deepcopy


def pil2cv(image):
    ''' PIL -> OpenCV '''
    new_image = np.array(image)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    return new_image


def cv2pil(image):
    ''' OpenCV -> PIL'''
    new_image = deepcopy(image)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:
        new_image = new_image[:, :, [2, 1, 0, 3]]
    new_image = Image.fromarray(new_image)
    return new_image


def imshow(image):
    image = pil2cv(image)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cam_test(model):
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        ret, frame = cap.read()
        frame = np.array(frame)

        ############################
        frame = model.compute(frame)
        ############################

        cv2.imshow('demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def load_image(path_image):
    frame = np.array(Image.open(path_image))
    return frame


def l2_normalization(v):
    l2_norm = np.linalg.norm(v, ord=2)
    v_l2_norm = v / l2_norm
    return v_l2_norm 


def fig2cv(fig):
    fig.canvas.draw()
    # im = np.array(fig.canvas.renderer.buffer_rgba())
    im = np.array(fig.canvas.renderer._renderer) # matplotlibが3.1より前の場合
    im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
    return im


def plot_3d_pose(pose_3d):
    if pose_3d.shape[0]>1:
        pose_3d = pose_3d[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], alpha=0.4, color='red')
    for body_edge in body_edges:
        ax.plot(
            (pose_3d[body_edge[0]][0], pose_3d[body_edge[1]][0]),
            (pose_3d[body_edge[0]][1], pose_3d[body_edge[1]][1]),
            (pose_3d[body_edge[0]][2], pose_3d[body_edge[1]][2]),
        color='skyblue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-100,100)
    ax.set_ylim(100,200)
    ax.set_ylim(-200,0)
    return fig


def scale_to_height(img, height):
    scale = height / img.shape[0]
    return cv2.resize(img, dsize=None, fx=scale, fy=scale)


def generate_canvas(xmin, ymin, xmax, ymax, ratio_frame=16/9):
    height_bbox = ymax - ymin
    width_bbox = xmax - xmin
    ratio_bbox = width_bbox / height_bbox

    if ratio_bbox < ratio_frame:
        width_canvas = int(height_bbox * ratio_frame)
        canvas_org = np.zeros((height_bbox, width_canvas, 3), np.uint8)
    elif ratio_bbox > ratio_frame:
        height_canvas = int(width_bbox / ratio_frame)
        canvas_org = np.zeros((height_canvas, width_bbox, 3), np.uint8)
    elif ratio_bbox == ratio_frame:
        canvas_org = np.zeros((height_bbox, width_bbox, 3), np.uint8)
    else:
        raise ValueError
    return canvas_org


def gen_nan_mat(dim, data_num=None):
    if data_num is None:
        mat = np.zeros(dim)
    else:
        mat = np.zeros((data_num, dim))
    mat[:] = np.nan
    return mat
