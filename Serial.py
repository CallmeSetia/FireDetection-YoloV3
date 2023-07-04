import threading
import sys
import signal

import serial
from datetime import datetime
import telebot

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

class TelegramClient:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.bot = self._create_bot()

    def _create_bot(self):
        bot = telebot.TeleBot(self.token)
        return bot

    def send_message(self, text):
        self.bot.send_message(self.chat_id, text)

    def send_imagecv2(self, image):
        # Konversi frame OpenCV menjadi data JPEG
        _, jpeg_image = cv2.imencode(".jpg", image)
        # Kirim gambar menggunakan sendPhoto
        self.bot.send_photo(self.chat_id, photo=jpeg_image.tobytes())

    def start_polling(self):
        self.bot.polling(none_stop=True, interval=0, timeout=20)

    def stop_polling(self):
        self.bot.stop_polling()


class SensorData:
    def __init__(self, data_list):
        self.latitude = data_list[0]
        self.longitude = data_list[1]
        self.temperature = data_list[2]
        self.humidity = data_list[3]
        self.smoke_sensor = data_list[4]


class SensorDataReader:
    def __init__(self, port='COM6', baud_rate=9600, timeout=1, buffer_size=5, process=None):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.buffer_size = buffer_size
        self.serial_conn = None
        self.buffer = []
        self.process = process
        self.data = None

    def open_serial_connection(self):
        try:
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
            self.serial_conn.reset_input_buffer()
            print("[INFO] Serial Communication Initialized.")
        except Exception as err:
            print(f"[ERROR] Failed to initialize serial communication: {err}")

    def close_serial_connection(self):
        try:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                print("[INFO] Serial Connection closed.")
        except Exception as err:
            print(f"[ERROR] Error closing serial connection: {err}")

    def read_data(self):
        try:
            if self.serial_conn and self.serial_conn.is_open:
                buffer_data = self.serial_conn.readline().decode('utf-8', 'ignore').strip().split('#')
                if len(buffer_data) > 3:
                    self.buffer.append(buffer_data)
                    self.data = self.buffer[0]
                    if len(self.buffer) > self.buffer_size:
                        self.buffer.pop(0)

        except KeyboardInterrupt:
            print("[INFO] Keyboard Interrupt received. Closing the serial connection safely.")
            self.close_serial_connection()

        except serial.SerialException:
            print("[ERROR] Serial port disconnected. Closing the serial connection safely.")
            self.close_serial_connection()

    def get_data(self):
        try:
            return self.data
        except KeyboardInterrupt:
            print("[INFO] Keyboard Interrupt received. Closing the serial connection safely.")
            self.close_serial_connection()

        except serial.SerialException:
            print("[ERROR] Serial port disconnected. Closing the serial connection safely.")
            self.close_serial_connection()
# Fungsi untuk menghentikan program secara aman saat Ctrl+C atau terminal ditutup
def signal_handler(signal, frame):
    print("Program dihentikan")
    sys.exit(0)

# Pengaturan sinyal untuk mengatasi Ctrl+C atau terminal ditutup
signal.signal(signal.SIGINT, signal_handler)

DataSensor = ''
DeteksiClass = ''
DeteksiFrame = ''
def arduino_handler():
    global DataSensor

    sensor_reader = SensorDataReader(port='COM6', baud_rate=9600)
    sensor_reader.open_serial_connection()

    while True:
        sensor_reader.read_data()
        Data = sensor_reader.get_data()

        DataSensor = Data
        if Data:
            sensor = SensorData(data_list=Data)
            message = "[INFO] New data received: " \
                      "Latitude={}, " \
                      "Longitude={}, " \
                      "Temperature={}," \
                      " Humidity={}, " \
                      "Smoke Sensor={}".format(
                sensor.latitude,
                sensor.longitude,
                sensor.temperature,
                sensor.humidity,
                sensor.smoke_sensor
            )

            print(message)

def telegram_handler():
    global DeteksiClass, DeteksiFrame, DataSensor

    telegram_token = "6322020435:AAELvBH9XaWxsG5hFZ4vn-FTm70B69wsB20"
    chat_id = "678809573"

    current_time = datetime.now().strftime("%H:%M:%S")
    current_date = datetime.now().strftime("%Y-%m-%d")

    telegram_client = TelegramClient(telegram_token, chat_id)

    sensor = ''
    last_print_time = 0
    print_time = 5 # detik
    while True:
        if DataSensor is not None and len(DataSensor) > 3:
            sensor = SensorData(data_list=DataSensor)
            if "api" in DeteksiClass:
                current_time = time.time()
                if current_time - last_print_time >= print_time:
                    telegram_client.send_imagecv2(DeteksiFrame)
                    telegram_client.send_message(f"[INFO] Ada Api pada {current_date} {current_time}")
                    telegram_client.send_message(
                        "[INFO] Status Sensor Temperature={}, Humidity={}, Smoke Sensor={}".format(sensor.temperature,sensor.humidity,sensor.smoke_sensor)
                                                )
                    telegram_client.send_message(
                        "[INFO] Lokasi :  https://www.google.com/maps/search/?api=1&query={},{}" .format(sensor.latitude, sensor.longitude)
                    )

                    last_print_time = current_time
            else:
                if float(sensor.temperature) > 50:
                    current_time = time.time()
                    if current_time - last_print_time >= print_time:
                        telegram_client.send_imagecv2(DeteksiFrame)
                        telegram_client.send_message(f"[INFO] Ada Api pada {current_date} {current_time}")
                        telegram_client.send_message(
                            "[INFO] Status Sensor Temperature={}, Humidity={}, Smoke Sensor={}"
                            .format(sensor.temperature, sensor.humidity, sensor.smoke_sensor)
                        )
                        telegram_client.send_message(
                            "[INFO] Lokasi :  https://www.google.com/maps/search/?api=1&query={},{}"
                            .format(sensor.latitude, sensor.longitude)
                        )

                    last_print_time = current_time
                else: # Tidak Ada Api
                    current_time = time.time()
                    if current_time - last_print_time >= print_time:
                        print("tidak ada api", sensor.temperature, sensor.humidity)

                        last_print_time = current_time


    # message = ""
    # telegram_client.send_message(message)

def whatsapp_handler():
    pass


def deteksi_handler():
    global DeteksiClass, DeteksiFrame

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
            frame = resize(frame, 640, 640)
            # ksize
            ksize = (3, 3)

            # Using cv2.blur() method
            frame = cv2.blur(frame, ksize)
            detect = yolo.predict(frame)

            DeteksiClass = [data['class'] for data in detect]
            DeteksiFrame = frame

            yolo.draw(frame, detect)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1500) == ord('q'):
                break

    except Exception as e:
        print(e)


if __name__ == "__main__":
    try:
        Arduino = threading.Thread(target=arduino_handler)
        Arduino.daemon = True
        Arduino.start()

        Telegram = threading.Thread(target=telegram_handler)
        Telegram.daemon = True
        Telegram.start()

        Image = threading.Thread(target=deteksi_handler)
        Image.daemon = True
        Image.start()

    except KeyboardInterrupt:
        # Tangkap exception jika pengguna menekan Ctrl+C
        print("KeyboardInterrupt: Stopping the program")


    except (RuntimeError, Exception) as e:
        print(f"[ERROR] {datetime.timestamp(datetime.now())} Serial Initialize Failed: \n{e}")

    finally:
        Arduino.join()
        Telegram.join()
        Image.join()


