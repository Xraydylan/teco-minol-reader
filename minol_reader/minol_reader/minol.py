from ipaddress import IPv4Address
from .ESP32_CAM import ESP_CAM
from .ssocr import SSOCR
from .utils import Ping
import sys
import cv2
import time
import numpy as np
from pathlib import Path
from numbers import Number

target_frame_shape = (720, 1280, 3)

cam_config_default = {
    "default": {
        "framesize": 11,
        "contrast": 2,
        "saturation": 2,
        "special_effect": 2,
        "aec": 0,
        "aec_value": 400  # 600; max 1200
    },
    "led_config": {
        "max_intensity": 100,
        "default_intensity": 50  # 60
    }
}

cam_timing_default = {
    "ESP_CAM_timeout": 5,
    "ESP_CAM_delay": 0.01
}

threshold_limits = (20, 35)


class MINOL:
    def __init__(self, cam_ip: str, cam_config: dict = None, cam_timing: dict = None, decimal_position: int = None,
                 save_images: bool = None, monotone_detection: bool = None, exit_on_error: bool = None,
                 image_parameters: dict = None):
        if cam_config is None:
            cam_config = cam_config_default
        if cam_timing is None:
            cam_timing = cam_timing_default
        if decimal_position is None:
            decimal_position = 3
        if save_images is None:
            save_images = False
        if monotone_detection is None:
            monotone_detection = save_images
        if exit_on_error is None:
            exit_on_error = False

        self.save_images = save_images
        self.monotone_detection = monotone_detection and save_images
        self.exit_on_error = exit_on_error

        self.decimal_position = decimal_position

        self.cam_ip = IPv4Address(cam_ip)
        self.ping = Ping([self.cam_ip])
        self.cam = ESP_CAM(self.cam_ip, "ESP32_CAM", cam_config, cam_timing)
        self.ssocr = SSOCR(debug=False)

        self.max_gray_ratio = 0.5  # previously 0.7

        self.rough_cut_indices = (0, 450, 0, 1050)  # Old

        # Display cut Defaults
        self.fcut_1 = (400, 316)  # upper left
        self.fcut_2 = (854, 426)  # lower right

        cut_indices = image_parameters.get("cut_indices")
        if cut_indices is not None:
            self.fcut_1 = cut_indices.get("upper_left", self.fcut_1)
            self.fcut_2 = cut_indices.get("lower_right", self.fcut_2)

        # Calib Centers Defaults
        self.c1 = (151, 499)
        self.c2 = (1175, 644)

        calib_centers = image_parameters.get("calib_centers")
        if calib_centers is not None:
            self.c1 = calib_centers.get("c1", self.c1)
            self.c2 = calib_centers.get("c2", self.c2)

        # Calib areas Defaults
        self.a1 = ((100, 460), (200, 554))
        self.a2 = ((1120, 570), (1243, 701))

        calib_areas = image_parameters.get("calib_areas")
        if calib_areas is not None:
            dict_a1 = calib_areas.get("a1")
            dict_a2 = calib_areas.get("a2")
            self.a1 = (dict_a1.get("upper_left", self.a1[0]), dict_a1.get("lower_right", self.a1[1]))
            self.a2 = (dict_a2.get("upper_left", self.a2[0]), dict_a2.get("lower_right", self.a2[1]))

        # Advanced 2 point calibration NOT IMPLEMENTED yet!!!
        # self.only_simple_translation = image_parameters.get("only_simple_translation", True)
        self.only_simple_translation = True

        self.skip_position_calibration = image_parameters.get("skip_position_calibration", False)

        self.x_skew_factor = image_parameters.get("x_skew_factor", -0.06)

        # May be adjusted
        self.bi_end_left = image_parameters.get("bi_end_left", 125)
        self.bi_start_right = image_parameters.get("bi_start_right", 260)
        self.bi_max_left = image_parameters.get("bi_max_left", 75)
        self.bi_max_right = image_parameters.get("bi_max_right", 50)

        self.contrast_factor = image_parameters.get("contrast_factor", 2)

        self._count = 0

        self.retries = 4

        self._last_value: Number = 0

    def start(self):
        print("Starting")
        self.wait_for_online()
        time.sleep(1)

    def wait_for_online(self):
        print("Waiting for ESP")
        while 1:
            if self.is_online():
                break
            else:
                print(".", end="")
        print("ESP ONLINE!")

    def is_online(self):
        return self.ping.ping_next()[1]

    def show_frame(self):
        print("Connecting!")
        self.cam.set_online_status(True, wait=True, affect_worker=False)
        time.sleep(1)
        print("Selecting Camera")
        self.cam.select()
        print("Stabilizing")
        time.sleep(6)

        print("Fetching Frame")
        frame = self._fetch_frame()

        print("Deselecting Camera")
        self.cam.deselect()

        print("Disconnecting")
        self.cam.set_online_status(False, wait=True, affect_worker=False)

        cv2.imshow("Frame", frame)
        cv2.setMouseCallback("Frame", self._click_callback)
        self._smart_key_wait("Frame", delay=0.5)

        fcut_1, fcut_2 = self.fcut_1, self.fcut_2
        if not self.skip_position_calibration:
            # Corrects position of cut
            f1cor, f2_cor = self._position_calibration(frame)
            fcut_1 = (fcut_1[0] + f1cor[0], fcut_1[1] + f1cor[1])
            fcut_2 = (fcut_2[0] + f2_cor[0], fcut_2[1] + f2_cor[1])

        frame_cut = self._cut(frame, fcut_1, fcut_2)

        cv2.imshow("Frame_Cut", frame_cut)
        self._smart_key_wait("Frame_Cut", delay=0.5)

        frame_normal = self._skew_image(frame_cut)

        frame_gray = cv2.cvtColor(frame_normal, cv2.COLOR_RGB2GRAY)

        print("Bi-Gradient")
        # 125, 260, 75, 50
        frame_normal = self._bi_gradient(frame_gray, self.bi_end_left, self.bi_start_right, self.bi_max_left,
                                         self.bi_max_left)

        print("Add contrast")
        # 2
        frame_normal = self._contrast(frame_normal, self.contrast_factor)

        cv2.imshow("Frame_Cut_Final", frame_normal)
        self._smart_key_wait("Frame_Cut_Final", delay=0.5)

    def get_number(self):
        print("Connecting!")
        self.cam.set_online_status(True, wait=True, affect_worker=False)
        time.sleep(1)
        print("Selecting Camera")
        self.cam.select()
        print("Stabilizing")
        time.sleep(6)

        print("Fetching Frame")
        digit_list, frame = self._frame_to_digit_list()

        print("Converting to Number")
        number = self._get_number_from_frame(digit_list, frame)

        self._frame_saver_test(frame, number)
        self._last_value = self._last_value if number is None else number

        print("Deselecting Camera")
        self.cam.deselect()

        print("Disconnecting")
        self.cam.set_online_status(False, wait=True, affect_worker=False)
        return number

    def reselect(self):
        print("Reselecting")
        self.cam.deselect()
        self.cam.set_online_status(False, wait=True, affect_worker=False)
        time.sleep(2)
        self.cam.set_online_status(True, wait=True, affect_worker=True)
        time.sleep(2)
        self.cam.select()
        time.sleep(2)

    def _get_frame(self):
        return self.cam.get_frame()  # [online, frame]

    def get_frame(self):
        print("Connecting!")
        self.cam.set_online_status(True, wait=True, affect_worker=True)
        time.sleep(2)
        print("Selecting Camera")
        self.cam.select()
        print("Stabilizing")
        time.sleep(5)

        frame = self._fetch_frame()

        print("Deselecting Camera")
        self.cam.deselect()

        print("Disconnecting")
        self.cam.set_online_status(False, wait=True, affect_worker=True)
        return frame

    # old
    def _rough_cut(self, frame):
        return frame[self.rough_cut_indices[0]:self.rough_cut_indices[1],
               self.rough_cut_indices[2]:self.rough_cut_indices[3]]

    def _cut(self, frame, p1, p2):
        # Implement better cutting according to rectangle
        # p1 -> upper left
        # p2 -> lower right
        indices = (p1[1], p2[1], p1[0], p2[0])
        return frame[indices[0]:indices[1], indices[2]:indices[3]]

    def _skew_image(self, frame):
        # Height and width of the image
        h, w = frame.shape[:2]

        distance = 100
        skew_value = distance * self.x_skew_factor

        # Source points - choose points on the skewed image
        src_points = np.float32([[0, 0],
                                 [0, distance],
                                 [distance, 0]])  # replace x1, y1, ..., y3 with actual values

        # Destination points - corresponding points in the unskewed image
        dst_points = np.float32([[skew_value, 0],
                                 [0, distance],
                                 [distance + skew_value, 0]])  # replace x1_prime, y1_prime, ..., y3_prime with values

        # Compute the affine matrix
        affine_matrix = cv2.getAffineTransform(src_points, dst_points)

        # Apply the transformation, make added boarder white
        corrected_image = cv2.warpAffine(frame, affine_matrix, (w, h), borderValue=(255, 255, 255))
        return corrected_image

    def _bi_gradient(self, image, end_left, start_right, max_left, max_right):
        # 125, 250, 50, 75

        # Gradients
        gradient_r = np.linspace(0, max_right, image.shape[1] - start_right).astype(np.uint8)
        gradient_l = np.linspace(0, max_left, end_left).astype(np.uint8)[::-1]

        # Convert image
        image_16 = image.astype(np.uint16)

        # Add gradients
        image_16[:, start_right:] += gradient_r
        image_16[:, :end_left] += gradient_l

        # Clamp to max value
        image_16[image_16 > 255] = 255

        # Convert back to uint8
        image = image_16.astype(np.uint8)
        return image

    def _contrast(self, image, factor):
        # Subtract min value
        image = image[:, :] - np.min(image)

        # Convert to uint16 and multiply with factor
        image = image.astype(np.uint16) * factor

        # Clamp to max value
        image[image > 255] = 255

        # Convert back to uint8
        image = image.astype(np.uint8)
        return image

    def _position_calibration(self, frame):
        # calib_points_1 = ((100, 460), (200, 554))
        # calib_points_2 = ((1120, 570), (1243, 701))

        calib_points_1 = self.a1
        calib_points_2 = self.a2

        p1 = self.find_point_cords(frame, calib_points_1)

        if p1 is None:
            return (0, 0), (0, 0)

        if not self.only_simple_translation:
            p2 = self.find_point_cords(frame, calib_points_2)
            if p2 is None:
                return (0, 0), (0, 0)

            # NOT IMPLEMENTED YET
            return (0, 0), (0, 0)

        # Make difference between p1 and c1
        return (p1[0] - self.c1[0], p1[1] - self.c1[1]), (p1[0] - self.c1[0], p1[1] - self.c1[1])




    def find_point_cords(self, frame, calib_points):
        region_1 = self._cut(frame, calib_points[0], calib_points[1])

        region_1 = cv2.cvtColor(region_1, cv2.COLOR_BGR2GRAY)

        region_1 = cv2.GaussianBlur(region_1, (3, 3), 0)

        threshold = 30
        blocksize = 155
        dst = cv2.adaptiveThreshold(region_1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize,
                                    threshold)

        circle = self.find_disk(dst)
        x, y, _ = circle
        x += calib_points[0][0]
        y += calib_points[0][1]
        return x, y  # x, y

    def find_disk(self, frame):
        # Reduce the noise to avoid false circle detection
        blurr = cv2.GaussianBlur(frame, (3, 3), 0)

        circles = cv2.HoughCircles(blurr, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=10, minRadius=2, maxRadius=30)

        if circles is None:
            print("No circles found!")
            return None

        if len(circles) > 1:
            print("More than one circle found!")
            return None

        circles = np.uint16(np.around(circles))
        return circles[0][0]  # x, y, r

    def _fetch_frame(self):
        while 1:
            online, frame = self._get_frame()

            if not online:
                print("Camera offline?")

                if self.exit_on_error:
                    print("Exiting")
                    sys.exit(1)

                self.reselect()  # reselecting here may not work
                continue

            if not self._frame_sanity_check(frame):
                time.sleep(1)
                continue

            ratio = self.black_ratio(frame)
            # print(ratio)
            if ratio < self.max_gray_ratio:
                break
            time.sleep(0.3)
        return frame

    def _frame_to_digit_list(self, threshold=None):
        frame = self._fetch_frame()

        fcut_1, fcut_2 = self.fcut_1, self.fcut_2
        if not self.skip_position_calibration:
            # Corrects position of cut
            print("Position Calibration")
            f1cor, f2_cor = self._position_calibration(frame)
            fcut_1 = (fcut_1[0] + f1cor[0], fcut_1[1] + f1cor[1])
            fcut_2 = (fcut_2[0] + f2_cor[0], fcut_2[1] + f2_cor[1])

        print("Fine Cutting")
        frame = self._cut(frame, fcut_1, fcut_2)

        print("Skew normalizing")
        frame = self._skew_image(frame)

        print("Convert to grey")
        # Convert to grey
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        print("Bi-Gradient")
        # 125, 260, 75, 50
        frame = self._bi_gradient(frame, self.bi_end_left, self.bi_start_right, self.bi_max_left, self.bi_max_left)

        print("Add contrast")
        # 2
        frame = self._contrast(frame, self.contrast_factor)

        return self._extract_digits_list(frame, threshold=threshold)

    def _get_number_from_frame(self, digit_list, frame):
        number = self.ssocr.digits_to_number(digit_list, decimal_position=self.decimal_position)

        if number is None:
            print("Couldn't convert digits to number")
            step = (threshold_limits[1] - threshold_limits[0]) / self.retries
            for threshold in np.arange(*threshold_limits, step):
                print(f"Trying threshold: {threshold}")
                digit_list, _ = self._extract_digits_list(frame, threshold=threshold)
                number = self.ssocr.digits_to_number(digit_list, decimal_position=self.decimal_position)
                if number is not None:
                    break
        return number

    def _extract_digits_list(self, frame, threshold: int = None):
        try:
            # Makes sure image is already grey-scaled
            digit_list = self.ssocr.digits_from_image(frame, convert_grey=False, threshold=threshold)
        except Exception as e:
            print("Couldn't recognize digits: ", e)
            print("Retrying")
            time.sleep(5)
            return self._frame_to_digit_list()
        return digit_list, frame

    def _click_callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"X: {x}, Y: {y}")

    def _smart_key_wait(self, window: str, delay: Number = None):
        # delay in seconds

        if delay is None:
            cv2.waitKey()
            cv2.destroyAllWindows()
            return

        wait_time = int(1000 * delay)
        while cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1:
            keyCode = cv2.waitKey(wait_time)
            if keyCode != -1:
                cv2.destroyAllWindows()
                return

    def _frame_sanity_check(self, frame):
        if frame is None:
            print("Frame is None")
            return False
        if frame.shape != target_frame_shape:
            print("Frame shape is not correct")
            print(f"{frame.shape} != {target_frame_shape}")
            self.reselect()
            return False
        return True

    def _frame_saver_test(self, frame, number):
        # Saves image with number as name
        if not self.save_images:
            return

        folder = Path("output")
        folder.mkdir(parents=True, exist_ok=True)

        number_str = str(number).replace(".", "-") if number is not None else "None"
        image_name = f"image_{self._count}_{number_str}.png"
        file_path = folder / image_name
        cv2.imwrite(str(file_path), frame)
        self._count += 1

        if not self.monotone_detection:
            return

        if number is None:
            return

        if number >= self._last_value:
            return

        # Number got smaller
        # Output folder does exist
        monotone_path = folder / "monotone.txt"
        with monotone_path.open("a") as f:
            f.write(f"{self._count - 1}: {number} < {self._last_value}\n")

    def apply_threshold(self, image_array, threshold):
        # Create a copy of the original array to avoid modifying the original data
        thresholded_image = np.copy(image_array)

        # Apply the threshold
        thresholded_image[thresholded_image < threshold] = 0
        thresholded_image[thresholded_image >= threshold] = 255
        return thresholded_image

    def black_ratio(self, frame, convert_grey: bool = True):
        if convert_grey:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return np.sum(frame < 100) / frame.size
