import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def make_video(x, y, filename):
    rotation_matrix = R.from_euler('Z', -90, degrees=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videowriter = cv2.VideoWriter(filename, fourcc, 25.0, (720, 360), True)
    scale = 10**(np.log10(np.max(y)).astype(int))
    x = x / scale
    y = y / scale
    try:
        for (pts_x, pts_y) in tqdm(zip(x, y), desc=f'{filename}', total=x.shape[0]):
            pts_xleft = rotation_matrix.apply(pts_x-[0, 1.5, 0]) + [1, 1.5, 0]      # left view
            pts_yleft = rotation_matrix.apply(pts_y-[0, 1.5, 0]) + [1, 1.5, 0]
            
            pts_x = pts_x + [-1, 0, 0]                                              # front view
            pts_y = pts_y + [-1, 0, 0]

            pts_x = np.concatenate((pts_x, pts_xleft))
            pts_y = np.concatenate((pts_y, pts_yleft))

            pts_x = (pts_x[:, (0, 2)] * [180, -180] + [360, 180]).astype(int)
            pts_y = (pts_y[:, (0, 2)] * [180, -180] + [360, 180]).astype(int)

            image = np.zeros((360, 720, 3), np.uint8)
            for (ox, oy) in pts_x:
                ox, oy = int(ox), int(oy)
                cv2.circle(image, (ox, oy), 3, (0, 0, 255), -1)
            for (ox, oy) in pts_y:
                ox, oy = int(ox), int(oy)
                cv2.circle(image, (ox, oy), 3, (0, 255, 0), -1)

            videowriter.write(image)
    except KeyboardInterrupt:
        videowriter.close()
