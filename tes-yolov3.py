import os
import cv2
import numpy as np
import urllib.request
import time

class Vision:
    def __init__(self, isUsingCam=None, addr=None, index=0):
        # write configuration
        self.frame_count = 0
        self.filenames = None
        self.fourcc = None
        self.out = None

        # get address
        self.cap = None
        self.success = False
        self.index = index
        if isUsingCam:
            while not self.success:
                try:
                    print("[INFO] Initialize Camera")
                    self.cap = cv2.VideoCapture(self.index)
                    if not self.cap.isOpened():
                        raise Exception(f"Cannot Open Camera by Index {self.index}")
                    ret, frame = self.cap.read()
                    if not ret:
                        raise Exception(f"Failed to Capture Frame by Index {self.index}")
                    self.success = True
                except Exception as err:
                    print(f"[ERROR] Camera Initialization Failed: {err}")
                    time.sleep(0.5)
                    self.index += 1
            print(f"[INFO] Camera Initialization Success")
        else:
            self.cap = cv2.VideoCapture(addr)

        # fps
        self._prev_time = 0
        self._new_time = 0

    # def writeConfig(self, name="output.mp4", types="mp4v"):  # XVID -> avi
    #     self.filenames = name
    #     self.fourcc = cv2.VideoWriter_fourcc(*types)  # format video
    #     # filename, format, FPS, frame size
    #     self.out = cv2.VideoWriter(
    #         self.filenames, self.fourcc, 15.0, (450, 337))

    def write(self, frame):
        self.out.write(frame)

    def writeImg(self, frame, path="cats-output.png"):
        filename = path
        cv2.imwrite(filename, frame)
        with open(filename, 'ab') as f:
            f.flush()
            os.fsync(f.fileno())

    def resize(self, image, width=None, height=None,
               interpolation=cv2.INTER_AREA):
        dim = None
        w = image.shape[1]
        h = image.shape[0]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation=interpolation)
        return resized

    def __get_fps(self):
        fps = 0.0
        try:
            self._new_time = time.time()
            fps = 1 / (self._new_time - self._prev_time)
            self._prev_time = self._new_time
            fps = 30 if fps > 30 else 0 if fps < 0 else fps
        except ZeroDivisionError as e:
            pass
        return int(fps)

    def blur(self, frame=None, sigma=11):
        return cv2.GaussianBlur(frame, (sigma, sigma), 0)

    def setBrightness(self, frame, value):
        h, s, v = cv2.split(
            cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        v = np.clip(v.astype(int) + value, 0, 255).astype(np.uint8)
        return cv2.cvtColor(
            cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    def setContrast(self, frame, value):
        alpha = float(131 * (value + 127)) / (127 * (131 - value))
        gamma = 127 * (1 - alpha)
        return cv2.addWeighted(
            frame, alpha, frame, 0, gamma)

    def setBrightnessNcontrast(self, frame, bright=0.0, contr=0.0, beta=0.0):
        return cv2.addWeighted(frame, 1 + float(contr)
                               / 100.0, frame, beta, float(bright))

    def read(self, frame_size=480, show_fps=False):
        try:
            success, frame = self.cap.read()
            if not success:
                raise RuntimeError
            if show_fps:
                try:  # put fps
                    cv2.putText(frame, str(self.__get_fps()) + " fps", (20, 40), 0, 1,
                                [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                except RuntimeError as e:
                    print(e)
            frame = self.resize(frame, frame_size)
            return frame
        except RuntimeError as e:
            print("[INFO] Failed to capture the Frame")

    def readFromUrl(self, url="http://192.168.200.24/cam-hi.jpg", frame_size=480, show_fps=False):
        try:
            img_resp = urllib.request.urlopen(url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(imgnp, -1)
            if show_fps:
                try:  # put fps
                    cv2.putText(frame, str(self.__get_fps()) + " fps", (20, 40), 0, 1,
                                [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                except RuntimeError as e:
                    print(e)
            frame = self.resize(frame, frame_size)
            return frame
        except RuntimeError as e:
            print("[INFO] Failed to capture the Frame")

    def show(self, frame, winName="frame"):
        cv2.imshow(winName, frame)

    def wait(self, delay):
        return cv2.waitKey(delay)

    def release(self):
        self.cap.release()

    def destroy(self):
        cv2.destroyAllWindows()
class ImgRex:  # 3
    def __init__(self):
        pass

    def __map(self, x, inMin, inMax, outMin, outMax):
        return (x - inMin) * (outMax - outMin) // (inMax - inMin) + outMin

    def load(self, weight_path, cfg, classes):
        self.classes = None
        with open(classes, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.colors = np.random.uniform(
            0, 255, size=(len(self.classes), 3))  # optional
        self.net = cv2.dnn.readNet(weight_path, cfg)
        # Use DNN_BACKEND_CUDA and DNN_TARGET_CUDA for GPU support
        self.net = cv2.dnn.readNet(weight_path, cfg)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # self.net = cv2.dnn.readNetFromDarknet(cfg, weight_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1]
                              for i in self.net.getUnconnectedOutLayers()]
        # self.output_layers = self.net.getUnconnectedOutLayersNames()

    @staticmethod
    def draw(frame, detection):
        if detection is not []:
            for idx in detection:
                color = idx["color"]
                cv2.rectangle(
                    frame, (idx["x"], idx["y"]), (idx["x"] + idx["width"], idx["y"] + idx["height"]), color, 2)
                tl = round(0.002 * (frame.shape[0] + frame.shape[1]) / 2) + 1
                c1, c2 = (int(idx["x"]), int(idx["y"])), (int(
                    idx["width"]), int(idx["height"]))

                tf = int(max(tl - 1, 1))  # font thickness
                t_size = cv2.getTextSize(
                    idx["class"], 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

                cv2.rectangle(frame, c1, c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(frame, idx["class"] + " " + str(int(idx["confidence"] * 100)) + "%",
                            (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                cv2.circle(frame, (
                    int(idx["x"] + int(idx["width"] / 2)), int(idx["y"] + int(idx["height"] / 2))),
                           4, color, -1)
                cv2.putText(frame, str(int(idx["x"] + int(idx["width"] / 2))) + ", " + str(
                    int(idx["y"] + int(idx["height"] / 2))), (
                                int(idx["x"] + int(idx["width"] / 2) + 10),
                                int(idx["y"] + int(idx["height"] / 2) + 10)), cv2.FONT_HERSHEY_PLAIN, tl / 2,
                            [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return frame

    def predict(self, frame):
        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        height, width, ch = frame.shape
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        class_ids = []
        confidences = []
        boxes = []
        center = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # print(class_id, confidence)
                # if confidence > -1:
                    # object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                center.append([center_x, center_y])
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        values = []
        indexes = cv2.dnn.NMSBoxes(
            boxes, confidences, 0.0, 0.0)  # 0.4 changeable
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                label = str(self.classes[class_ids[i]])
                x, y, w, h = boxes[i]
                temp = {
                    "class": label,
                    "confidence": confidences[i],
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "center": center[i],
                    "color": self.colors[class_ids[i]]
                }
                values.append(temp)
        return values

def enhance_fire_color(image):
    # Peningkatan ketajaman dengan filter unsharp
    blurred = cv2.GaussianBlur(image, (0, 0), 10)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    # Peningkatan kontras dengan faktor 1.5
    enhanced_image = cv2.convertScaleAbs(sharpened, alpha=1.5, beta=0)

    # Peningkatan ketajaman dengan filter high-pass
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)
    return enhanced_image

def display_frames_in_folders(root_folder):
    # Mengecek apakah folder root ada
    if not os.path.exists(root_folder):
        print(f"Folder '{root_folder}' tidak ditemukan.")
        return

    # Mengiterasi semua folder di dalam root folder
    for foldername in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, foldername)
        # Mengecek apakah ini adalah folder
        if os.path.isdir(folder_path):
            # Mengecek apakah ada file gambar di dalam folder
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(image_files) == 0:
                print(f"Tidak ada file gambar di dalam folder '{foldername}'.")
                continue

            # Mengiterasi semua file gambar di dalam folder
            for filename in image_files:
                image_path = os.path.join(folder_path, filename)
                # Membaca dan menampilkan frame
                frame = cv2.imread(image_path)
                cv2.imshow('Frame', frame)
                # Menunggu 2 detik sebelum menampilkan frame berikutnya
                cv2.waitKey(2000)

    # Menutup jendela setelah semua folder dan frame ditampilkan
def convertScale(img, alpha, beta):
    """Add bias and gain to an image with saturation arithmetics. Unlike
    cv2.convertScaleAbs, it does not take an absolute value, which would lead to
    nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
    becomes 78 with OpenCV, when in fact it should become 0).
    """

    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)

# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = convertScale(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

def resize(image, width=None, height=None, interpolation=cv2.INTER_AREA):
        dim = None
        w = image.shape[1]
        h = image.shape[0]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation=interpolation)
        return resized

if __name__ == "__main__":
    # cam = Vision(isUsingCam=True, index=1)
    yolo = ImgRex()
    n = 0
    yolo.load("yolov3-tiny_training_final.weights", "tiny.cfg", "obj.names")
    root_folder = 'images'
    try:

            if not os.path.exists(root_folder):
                print(f"Folder '{root_folder}' tidak ditemukan.")

            # Mengiterasi semua folder di dalam root folder
            for foldername in os.listdir(root_folder):
                folder_path = os.path.join(root_folder, foldername)
                # print(folder_path)
                frame = cv2.imread(folder_path)
                frame = resize(frame, 640, 640 )
                # ksize
                ksize = (3, 3)

                # Using cv2.blur() method
                frame = cv2.blur(frame, ksize)
                detect = yolo.predict(frame)
                yolo.draw(frame, detect)
                cv2.imshow('frame', frame)
                # cam.show(frame, "frame")
                n = n + 1
                if n < 10:
                    if cv2.waitKey(500) == ord('q'):
                        break
                else :
                    if cv2.waitKey(500) == ord('q'):
                        break

    except Exception as e:
        print(e)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
