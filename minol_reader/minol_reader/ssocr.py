import cv2
import numpy as np
from pathlib import Path

DIGITS_LOOKUP = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (1, 1, 0, 0, 0, 0, 0): 1,
    (1, 0, 1, 1, 0, 1, 1): 2,
    (1, 1, 1, 0, 0, 1, 1): 3,
    (1, 1, 0, 0, 1, 0, 1): 4,
    (0, 1, 1, 0, 1, 1, 1): 5,
    (0, 1, 1, 1, 1, 1, 1): 6,
    (1, 1, 0, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 0, 1, 1, 1): 9,
    (0, 0, 0, 0, 0, 1, 1): '-'
}

DIGITS_LOOKUP_Bottom_OFF = {
    #    0  1  2  3  4  5  6
    (1, 1, 0, 1, 1, 1, 0): 0,
    (1, 1, 0, 0, 0, 0, 0): 1,
    (1, 0, 0, 1, 0, 1, 1): 2,
    (1, 1, 0, 0, 0, 1, 1): 3,
    (1, 1, 0, 0, 1, 0, 1): 4,
    (0, 1, 0, 0, 1, 1, 1): 5,
    (0, 1, 0, 1, 1, 1, 1): 6,
    (1, 1, 0, 0, 0, 1, 0): 7,
    (1, 1, 0, 1, 1, 1, 1): 8,
    (1, 1, 0, 0, 1, 1, 1): 9,
    (0, 0, 0, 0, 0, 1, 1): '-'
    #    0  1  2  3  4  5  6
}

# line positions
# 0:    Top Right
# 1:    Bottom Right
# 2:    Bottom
# 3:    Bottom Left
# 4:    Top Left
# 5:    Top
# 6:    Middle


H_W_Ratio = 1.9
# DEFAULT_THRESHOLD = 25  # 35
DEFAULT_THRESHOLD = 30  # 35
arc_tan_theta = 6.0


class SSOCR:
    def __init__(self, threshold: int = DEFAULT_THRESHOLD, bottom_off: bool = True, no_dot: bool = True, debug: bool = False):
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        if bottom_off is None:
            bottom_off = True
        if no_dot is None:
            no_dot = True
        if debug is None:
            debug = False


        self.threshold = threshold
        self.bottom_off = bottom_off
        self.no_dot = no_dot
        self.debug = debug

        self.not_identified = "*"

        self.kernel_size = (5, 5)

    def digits_from_file(self, file_path: str):
        img = self._load_image(file_path)
        return self.digits_from_image(img, convert_grey=False)

    def digits_from_image(self, image: np.ndarray, convert_grey: bool = True, threshold: int = None,
                          reserved_threshold: int = None, distance: int = None, separation: int = None):
        if threshold is None:
            threshold = self.threshold
        if reserved_threshold is None:
            reserved_threshold = 10
        if distance is None:
            distance = 10
        if separation is None:
            separation = 10

        dst = self._preprocess(image, convert_grey=convert_grey, threshold=threshold)
        digits_positions = self._find_digits_positions(dst, reserved_threshold=reserved_threshold, distance=distance, separation=separation)
        digits = self._recognize_digits_line_method(digits_positions, image, dst)

        if self.debug:
            print(digits)
            cv2.imshow('output', image)
            cv2.waitKey()
            cv2.destroyAllWindows()
        return digits

    def digits_to_number(self, digits: list, decimal_position: int = None):
        if decimal_position is None:
            decimal_position = 0  # default decimal position counted from the right

        if self.not_identified in digits:
            print(f"Not identified [{self.not_identified}] digits in list: {digits}")
            return None

        digits = digits.copy()
        digits.insert(len(digits) - decimal_position, '.')

        try:
            return float(''.join(map(str, digits)))
        except Exception as e:
            print("Couldn't convert digits to number: ", e)
            return None

    def _load_image(self, file_path: str):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError("File not found: " + file_path)
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    def _preprocess(self, image: np.ndarray, convert_grey: bool = None, threshold: int = None):
        if convert_grey is None:
            convert_grey = True
        if threshold is None:
            threshold = self.threshold

        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = image
        if convert_grey:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Camera images are RGB

        blurred = cv2.GaussianBlur(gray_image, (7, 7), 0)

        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
        img = clahe.apply(blurred)

        dst = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, threshold)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, self.kernel_size)
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
        dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)


        #rim_size = 10
        #height, width = dst.shape
        #rim = np.zeros((height + 2 * rim_size, width + 2 * rim_size), dtype=np.uint8)
        #rim[rim_size:rim_size + height, rim_size:rim_size + width] = dst

        if self.debug:
            cv2.imshow('equlizeHist', img)
            cv2.imshow('threshold', dst)
            #cv2.imshow('rim', rim)
        return dst

    def _find_digits_positions(self, img, reserved_threshold=10, distance=15, separation=10):
        digits_positions = []
        img_array = np.sum(img, axis=0)
        horizon_position = self._helper_extract(img_array, threshold=reserved_threshold, distance=distance, separation=separation)
        img_array = np.sum(img, axis=1)
        vertical_position = self._helper_extract(img_array, threshold=reserved_threshold * 4, distance=distance, separation=separation)
        # make vertical_position has only one element
        if len(vertical_position) > 1:
            vertical_position = [(vertical_position[0][0], vertical_position[len(vertical_position) - 1][1])]
        for h in horizon_position:
            for v in vertical_position:
                digits_positions.append(list(zip(h, v)))
        assert len(digits_positions) > 0, "Failed to find digits's positions"

        return digits_positions

    def _helper_extract(self, one_d_array, threshold=10, distance=15, separation=10):
        # original params threshold=20, distance=20, separation=12
        res = []
        flag = 0
        temp = 0
        separation_threshold = separation * 255
        for i in range(len(one_d_array)):
            if one_d_array[i] < separation_threshold:
                if flag > threshold:
                    start = i - flag
                    end = i
                    temp = end
                    if end - start > distance:
                        res.append((start, end))
                flag = 0
            else:
                flag += 1

        else:
            if flag > threshold:
                start = temp
                end = len(one_d_array)
                if end - start > 50:
                    res.append((start, end))
        return res

    def _recognize_digits_line_method(self, digits_positions, output_img, input_img):
        digits = []
        for c in digits_positions:
            x0, y0 = c[0]
            x1, y1 = c[1]
            roi = input_img[y0:y1, x0:x1]
            h, w = roi.shape
            suppose_W = max(1, int(h / H_W_Ratio))

            # 消除无关符号干扰
            if x1 - x0 < 25 and cv2.countNonZero(roi) / ((y1 - y0) * (x1 - x0)) < 0.2:
                continue

            # 对1的情况单独识别
            if w < suppose_W / 2:
                x0 = max(x0 + w - suppose_W, 0)
                roi = input_img[y0:y1, x0:x1]
                w = roi.shape[1]

            center_y = h // 2
            quater_y_1 = h // 4
            quater_y_3 = quater_y_1 * 3
            center_x = w // 2
            line_width = 5  # line's width
            width = (max(int(w * 0.15), 1) + max(int(h * 0.15), 1)) // 2
            small_delta = int(h / arc_tan_theta) // 4
            segments = [
                ((w - 2 * width, quater_y_1 - line_width), (w, quater_y_1 + line_width)),
                ((w - 2 * width, quater_y_3 - line_width), (w, quater_y_3 + line_width)),
                ((center_x - line_width - small_delta, h - 2 * width), (center_x - small_delta + line_width, h)),
                ((0, quater_y_3 - line_width), (2 * width, quater_y_3 + line_width)),
                ((0, quater_y_1 - line_width), (2 * width, quater_y_1 + line_width)),
                ((center_x - line_width, 0), (center_x + line_width, 2 * width)),
                ((center_x - line_width, center_y - line_width), (center_x + line_width, center_y + line_width)),
            ]
            on = [0] * len(segments)

            for (i, ((xa, ya), (xb, yb))) in enumerate(segments):
                #if ya < 2:
                #    ya += 10
                #    yb += 10
                seg_roi = roi[ya:yb, xa:xb]
                # plt.imshow(seg_roi, 'gray')
                # plt.show()
                total = cv2.countNonZero(seg_roi)
                area = (xb - xa) * (yb - ya) * 0.9
                # print('prob: ', total / float(area))
                if total / float(area) > 0.25:
                    on[i] = 1
            # print('encode: ', on)

            if self.bottom_off:
                on[2] = 0

            digit_dict = DIGITS_LOOKUP_Bottom_OFF if self.bottom_off else DIGITS_LOOKUP
            if tuple(on) in digit_dict.keys():
                digit = digit_dict[tuple(on)]
            else:
                digit = self.not_identified

            digits.append(digit)

            if not self.debug:
                continue

            if not self.no_dot:
                if cv2.countNonZero(roi[h - int(3 * width / 4):h, w - int(3 * width / 4):w]) / (
                        9. / 16 * width * width) > 0.65:
                    digits.append('.')
                    cv2.rectangle(output_img,
                                  (x0 + w - int(3 * width / 4), y0 + h - int(3 * width / 4)),
                                  (x1, y1), (0, 128, 0), 2)
                    cv2.putText(output_img, 'dot',
                                (x0 + w - int(3 * width / 4), y0 + h - int(3 * width / 4) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 0), 2)

            cv2.rectangle(output_img, (x0, y0), (x1, y1), (0, 128, 0), 2)
            cv2.putText(output_img, str(digit), (x0 + 3, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 0), 2)
        return digits


if __name__ == '__main__':
    test_folder = Path("/output")
    ssocr = SSOCR(debug=False)

    files = [file for file in test_folder.iterdir() if file.is_file()]

    key = lambda x: int(x.stem.split("_")[1])
    files.sort(key=key)

    for file in files:
        digits_from_file = ssocr.digits_from_file(str(file))
        print(f"{file.name}: {digits_from_file}")
        number = ssocr.digits_to_number(digits_from_file, 3)
        print(number, "\n")

