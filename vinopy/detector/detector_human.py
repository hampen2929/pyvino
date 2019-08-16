
import numpy as np
import math
import cv2

from vinopy.detector.detector import Detector


class DetectorObject(Detector):
    def __init__(self, task):
        self.task = task
        super().__init__(self.task)
    
    def get_box(self, bbox, frame):
        frame_h, frame_w = frame.shape[:2]
        box = bbox[3:7] * np.array([frame_w,
                                    frame_h,
                                    frame_w,
                                    frame_h])
        xmin, ymin, xmax, ymax = box.astype("int")
        self.xmin = xmin
        self.ymin = ymin
        return xmin, ymin, xmax, ymax

    def crop_bbox_frame(self, frame, xmin, ymin, xmax, ymax):
        bbox_frame = frame[ymin:ymax, xmin:xmax]
        return bbox_frame

    def get_frame_shape(self, frame):
        self.frame_h, self.frame_w = frame.shape[:2]
    
    def get_pos(self, frame):
        n, c, h, w = self.shapes
        self.frame_h, self.frame_w = frame.shape[:2]

        in_frame = self._in_frame(frame, n, c, h, w)
        self.exec_net.start_async(request_id=0, inputs={
                                  self.input_blob: in_frame})

        if self.exec_net.requests[0].wait(-1) == 0:
            res = self.exec_net.requests[0].outputs[self.out_blob]
            # prob threshold : 0.5
            bboxes = res[0][:, np.where(res[0][0][:, 2] > 0.5)]
        return bboxes

    def compute(self, init_frame, pred_flag=False, frame_flag=False):
        # copy frame to prevent from overdraw results
        frame = init_frame.copy()
        bboxes = self.get_pos(frame)
        results = {}
        for bbox_num, bbox in enumerate(bboxes[0][0]):
            xmin, ymin, xmax, ymax = self.get_box(bbox, frame)
            if pred_flag:            
                results[bbox_num] = {'label': bbox[1], 
                                     'conf': bbox[2], 
                                     'bbox': (xmin, ymin, xmax, ymax)}
            if frame_flag:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        if frame_flag:
            results = {'frame': frame}
        return results


class DetectorFace(DetectorObject):
    def __init__(self):
        self.task = 'detect_face'
        super().__init__(self.task)


class DetectorBody(DetectorObject):
    def __init__(self):
        self.task = 'detect_body'
        super().__init__(self.task)


class DetectorHeadpose(DetectorObject):
    def __init__(self):
        self.task = 'estimate_headpose'
        super().__init__(self.task)
        self.scale = 50
        self.focal_length = 950.0
        self.detector_face = DetectorFace()

    def _build_camera_matrix(self, center_of_face, focal_length):

        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        camera_matrix = np.zeros((3, 3), dtype='float32')
        camera_matrix[0][0] = focal_length
        camera_matrix[0][2] = cx
        camera_matrix[1][1] = focal_length
        camera_matrix[1][2] = cy
        camera_matrix[2][2] = 1

        return camera_matrix

    def _draw_axes(self, frame, center_of_face, yaw, pitch, roll, scale, focal_length):
        yaw *= np.pi / 180.0
        pitch *= np.pi / 180.0
        roll *= np.pi / 180.0

        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        Rx = np.array([[1,                0,                               0],
                       [0,                math.cos(pitch),  -math.sin(pitch)],
                       [0,                math.sin(pitch),   math.cos(pitch)]])
        Ry = np.array([[math.cos(yaw),    0,                  -math.sin(yaw)],
                       [0,                1,                               0],
                       [math.sin(yaw),    0,                   math.cos(yaw)]])
        Rz = np.array([[math.cos(roll),   -math.sin(roll),                 0],
                       [math.sin(roll),   math.cos(roll),                  0],
                       [0,                0,                               1]])

        # ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
        R = Rz @ Ry @ Rx
        
        camera_matrix = self._build_camera_matrix(center_of_face, focal_length)

        xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
        yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
        zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
        zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)

        o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
        o[2] = camera_matrix[0][0]

        xaxis = np.dot(R, xaxis) + o
        yaxis = np.dot(R, yaxis) + o
        zaxis = np.dot(R, zaxis) + o
        zaxis1 = np.dot(R, zaxis1) + o

        xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)

        xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)

        xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
        yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
        p1 = (int(xp1), int(yp1))
        xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))

        cv2.line(frame, p1, p2, (255, 0, 0), 2)
        cv2.circle(frame, p2, 3, (255, 0, 0), 2)

        return frame

    def get_axis(self, face_frame):
        n, c, h, w = self.shapes
        in_frame = self._in_frame(face_frame, n, c, h, w)
        self.exec_net.start_async(request_id=0, inputs={self.input_blob: in_frame})
        if self.exec_net.requests[0].wait(-1) == 0:
            yaw = .0  # Axis of rotation: y
            pitch = .0  # Axis of rotation: x
            roll = .0  # Axis of rotation: z
            # Each output contains one float value that represents value in Tait-Bryan angles (yaw, pitсh or roll).
            yaw = self.exec_net.requests[0].outputs['angle_y_fc'][0][0]
            pitch = self.exec_net.requests[0].outputs['angle_p_fc'][0][0]
            roll = self.exec_net.requests[0].outputs['angle_r_fc'][0][0]
        return yaw, pitch, roll

    def get_center_face(self, face_frame, xmin, ymin):
        n, c, h, w = self.shapes
        in_frame = self._in_frame(face_frame, n, c, h, w)
        self.exec_net.start_async(request_id=0, inputs={self.input_blob: in_frame})
        if self.exec_net.requests[0].wait(-1) == 0:
            center_of_face = (xmin + face_frame.shape[1] / 2, ymin + face_frame.shape[0] / 2, 0)
        return center_of_face

    def compute(self, init_frame, pred_flag=False, frame_flag=False):
        frame = init_frame.copy()
        faces = self.detector_face.get_pos(frame)
        results = {}
        for face_num, face in enumerate(faces[0][0]):
            xmin, ymin, xmax, ymax = self.get_box(face, frame)
            face_frame = frame[ymin:ymax, xmin:xmax]
            yaw, pitch, roll = self.get_axis(face_frame)
            center_of_face = self.get_center_face(face_frame, xmin, ymin)
            if (face_frame.shape[0] == 0) or (face_frame.shape[1] == 0):
                continue
            if pred_flag:
                results[face_num] = {'bbox': (xmin, ymin, xmax, ymax),
                                     'yaw': yaw, 'pitch': pitch, 'roll': roll, 
                                     'center_of_face': center_of_face}
            if frame_flag:
                scale = (face_frame.shape[0]**2 + face_frame.shape[1]**2)**0.5 / 2
                self._draw_axes(frame, center_of_face, yaw,
                                pitch, roll, scale, self.focal_length)
        if frame_flag:
            results['frame'] = frame
        return results


class DetectorEmotion(DetectorObject):
    def __init__(self):
        self.task = 'emotion_recognition'
        super().__init__(self.task)
        self.label = ('neutral', 'happy', 'sad', 'surprise', 'anger')
        self.detector_face = DetectorFace()

    def get_emotion(self, face_frame):
        n, c, h, w = self.shapes
        in_frame = self._in_frame(face_frame, n, c, h, w)
        self.exec_net.start_async(request_id=0, inputs={self.input_blob: in_frame})
        if self.exec_net.requests[0].wait(-1) == 0:
            res = self.exec_net.requests[0].outputs[self.out_blob]
            emotion = self.label[np.argmax(res[0])]
        return emotion

    def compute(self, init_frame, pred_flag=False, frame_flag=False, rect=True):
        assert isinstance(init_frame, np.ndarray)
        frame = init_frame.copy()
        faces = self.detector_face.get_pos(frame)
        results = {}
        for face_num, face in enumerate(faces[0][0]):
            xmin, ymin, xmax, ymax = self.get_box(face, frame)
            face_frame = self.crop_bbox_frame(frame, xmin, ymin, xmax, ymax)
            emotion = self.get_emotion(face_frame)
            if (face_frame.shape[0] == 0) or (face_frame.shape[1] == 0):
                continue
            if pred_flag:
                results[face_num] = {'bbox': (xmin, ymin, xmax, ymax),
                                     'emotion': emotion}
            if frame_flag:
                cv2.putText(frame, emotion,
                            (int(xmin + (xmax - xmin) / 2), int(ymin - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 2, cv2.LINE_AA)
                if rect:
                    frame = cv2.rectangle(frame,
                                            (xmin, ymin), (xmax, ymax),
                                            (0, 255, 0), 2)
        if frame_flag:
            results['frame'] = frame
        return results


class DetectorHumanPose(Detector):
    """
    https://github.com/opencv/opencv/blob/master/samples/dnn/openpose.py
    """
    def __init__(self):
        self.task = 'estimate_humanpose'
        super().__init__(self.task)
        self.thr_point = 0.1
        self.detector_body = DetectorBody()
        self.BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                           "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                           "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                           "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

        self.POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                           ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                           ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                           ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                           ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]
    
    def _get_heatmaps(self, frame):
        # frame include only one person
        n, c, h, w = self.shapes
        in_frame = self._in_frame(frame, n, c, h, w)
        self.exec_net.start_async(request_id=0, inputs={self.input_blob: in_frame})
        if self.exec_net.requests[0].wait(-1) == 0:
            res = self.exec_net.requests[0].outputs
        # pairwise_relations = res['Mconv7_stage2_L1']
        heatmaps = res['Mconv7_stage2_L2']
        return heatmaps

    def get_points(self, frame):
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        heatmaps = self._get_heatmaps(frame)
        points = []
        for num_parts in range(len(self.BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = heatmaps[0, num_parts, :, :]

            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frame_width * point[0]) / heatmaps.shape[3]
            y = (frame_height * point[1]) / heatmaps.shape[2]

            # Add a point if it's confidence is higher than threshold.
            # points.append((int(x), int(y)))
            points.append([int(x), int(y)] if conf > self.thr_point else [None, None])
        return points

    def draw_pose(self, init_frame, points):
        frame = init_frame.copy()
        for pair in self.POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert (partFrom in self.BODY_PARTS)
            assert (partTo in self.BODY_PARTS)

            idFrom = self.BODY_PARTS[partFrom]
            idTo = self.BODY_PARTS[partTo]

            if (points[idFrom][0] is not None) and (points[idTo][1] is not None):
                cv2.line(frame, tuple(points[idFrom]), tuple(points[idTo]), (0, 255, 0), 3)
                cv2.ellipse(frame, tuple(points[idFrom]), (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, tuple(points[idTo]), (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
        return frame

    def compute(self, frame, pred_flag=False, frame_flag=False):
        """ frame include multi person.

        Args:
            pred_flag: whether return pred results
            frame_flag: whether return frame

        Returns (dict): detected human pose points and drawn frame selectively.

        """
        bboxes = self.detector_body.get_pos(frame)
        results = {}
        for bbox_num, bbox in enumerate(bboxes[0][0]):
            xmin, ymin, xmax, ymax = self.detector_body.get_box(bbox, frame)
            bbox_frame = self.detector_body.crop_bbox_frame(frame, xmin, ymin, xmax, ymax)
            points = self.get_points(bbox_frame)
            if pred_flag:
                results[bbox_num] = {'points': points, 'bbox': (xmin, ymin, xmax, ymax)}
            if frame_flag:
                # TODO: draws points to original frame
                frame = self.draw_paris(frame, points)
        if frame_flag:
            results['frame'] = frame
        return results

    def compute_single(self, frame, pred_flag=False, frame_flag=False):
        # frame include only one person. Deteced person by DetectBody.
        results = {}
        points = self.get_points(frame)
        if pred_flag:
            results['points'] = points
        if frame_flag:
            frame = self.draw_pose(frame, points)
            results['frame'] = frame
        return results
