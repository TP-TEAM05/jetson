#!/usr/bin/env python3
import cv2
import numpy as np
import time
import argparse


class Config:
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30
    ROI_START = 0.55
    ROI_END = 0.95
    MIN_CONTOUR_AREA = 300
    MAX_CONTOUR_AREA = 80000
    KP = 0.40
    KI = 0.001
    KD = 0.08
    BASE_SPEED = 0.30
    MAX_SPEED = 0.50
    MIN_SPEED = 0.10
    MAX_STEER_DEG = 40
    FLIP_METHOD = 0


class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()

    def update(self, error):
        now = time.time()
        dt = max(now - self.prev_time, 0.001)
        self.integral = max(-1.0, min(1.0,
                            self.integral + error * dt))
        derivative = (error - self.prev_error) / dt
        output = (self.kp * error +
                  self.ki * self.integral +
                  self.kd * derivative)
        self.prev_error = error
        self.prev_time = now
        return output


class MotorController:
    def __init__(self, use_serial=False,
                 serial_port="/dev/ttyUSB0"):
        self.use_serial = use_serial
        self.ser = None
        if use_serial:
            import serial
            try:
                self.ser = serial.Serial(
                    serial_port, 115200, timeout=1)
                print(f"Serial pripojený: {serial_port}")
            except Exception as e:
                print(f"Serial nedostupný: {e}")
                self.use_serial = False

    def send(self, speed, steering_normalized):
        angle_deg = 90 + int(
            steering_normalized * Config.MAX_STEER_DEG)
        angle_deg = max(
            90 - Config.MAX_STEER_DEG,
            min(90 + Config.MAX_STEER_DEG, angle_deg))
        message = f"<{speed:.2f},{angle_deg}>"
        if self.use_serial and self.ser:
            try:
                self.ser.write(message.encode())
            except Exception as e:
                print(f"Serial chyba: {e}")
        return message

    def stop(self):
        self.send(0.0, 0.0)

    def cleanup(self):
        self.stop()
        if self.ser:
            self.ser.close()


class CSICamera:
    def __init__(self):
        self.cap = None

    def start(self):
        pipeline = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), "
            f"width={Config.FRAME_WIDTH}, "
            f"height={Config.FRAME_HEIGHT}, "
            f"format=NV12, framerate={Config.FPS}/1 ! "
            f"nvvidconv flip-method={Config.FLIP_METHOD} ! "
            f"video/x-raw, "
            f"width={Config.FRAME_WIDTH}, "
            f"height={Config.FRAME_HEIGHT}, "
            f"format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink drop=1"
        )
        self.cap = cv2.VideoCapture(
            pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(
                "CSI kamera sa nedá otvoriť!\n"
                "Skontroluj zapojenie FPC kábla.")
        print(f"CSI kamera OK: "
              f"{Config.FRAME_WIDTH}x"
              f"{Config.FRAME_HEIGHT}@{Config.FPS}fps")

    def get_frame(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def stop(self):
        if self.cap:
            self.cap.release()


class LineDetector:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_roi(self, frame):
        h = frame.shape[0]
        y_start = int(h * self.cfg.ROI_START)
        y_end = int(h * self.cfg.ROI_END)
        return frame[y_start:y_end, :], y_start

    def preprocess(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        # Normalizácia — vyrovná rozdiely v osvetlení
        normalized = cv2.normalize(
            blurred, None, 0, 255, cv2.NORM_MINMAX)

        # Adaptívny threshold — nevyžaduje pevné nastavenie
        # Automaticky sa prispôsobí osvetleniu
        # Hľadá pixely svetlejšie ako ich okolie = biela čiara
        binary = cv2.adaptiveThreshold(
            normalized, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, -10)

        # Väčší kernel — toleruje širšiu pásku (5-6cm)
        kernel = np.ones((7, 7), np.uint8)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel)
        return binary

    def find_center(self, binary):
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Filtruj šum a príliš veľké plochy
        valid_contours = [
            c for c in contours
            if self.cfg.MIN_CONTOUR_AREA
            < cv2.contourArea(c)
            < self.cfg.MAX_CONTOUR_AREA
        ]
        if not valid_contours:
            return None

        largest = max(valid_contours, key=cv2.contourArea)

        # Bounding box — stred pásky
        x, y, w, h = cv2.boundingRect(largest)
        cx_bbox = x + w // 2

        # Ťažisko
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx_moment = int(M["m10"] / M["m00"])
        cy_moment = int(M["m01"] / M["m00"])

        # Kombinuj pre presnejší stred širokej pásky
        cx = (cx_bbox + cx_moment) // 2
        return cx, cy_moment, w

    def detect(self, frame):
        roi, y_offset = self.get_roi(frame)
        binary = self.preprocess(roi)
        result = self.find_center(binary)
        frame_center = binary.shape[1] // 2
        debug = frame.copy()

        if result is None:
            # Červený obdĺžnik = čiara stratená
            cv2.rectangle(debug,
                          (0, y_offset),
                          (frame.shape[1],
                           int(frame.shape[0] * self.cfg.ROI_END)),
                          (0, 0, 255), 2)
            cv2.putText(debug, "CIARA STRATENA",
                        (10, y_offset - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            return None, debug

        cx, cy, width = result
        error = (cx - frame_center) / frame_center

        # Vizualizácia
        cv2.circle(debug,
                   (cx, cy + y_offset), 10,
                   (0, 0, 255), -1)
        cv2.line(debug,
                 (frame_center, y_offset),
                 (frame_center, frame.shape[0]),
                 (255, 0, 0), 2)
        cv2.line(debug,
                 (cx, y_offset),
                 (cx, frame.shape[0]),
                 (0, 255, 0), 2)
        cv2.rectangle(debug,
                      (0, y_offset),
                      (frame.shape[1],
                       int(frame.shape[0] * self.cfg.ROI_END)),
                      (255, 255, 0), 2)
        cv2.putText(debug,
                    f"Sirka: {width}px  Error: {error:+.2f}",
                    (10, y_offset - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2)
        return error, debug


class LineFollower:
    def __init__(self, show_preview=True,
                 use_serial=False,
                 serial_port="/dev/ttyUSB0"):
        self.cfg = Config()
        self.show_preview = show_preview
        self.pid = PID(self.cfg.KP,
                       self.cfg.KI,
                       self.cfg.KD)
        self.detector = LineDetector(self.cfg)
        self.motor = MotorController(use_serial, serial_port)
        self.camera = CSICamera()
        self.frame_count = 0
        self.lost_count = 0
        self.start_time = time.time()
        self.last_error = 0.0
        self.last_seen = time.time()

    def compute_speed(self, error):
        speed = self.cfg.BASE_SPEED * (1.0 - 0.5 * abs(error))
        return max(self.cfg.MIN_SPEED,
                   min(self.cfg.MAX_SPEED, speed))

    def run(self):
        self.camera.start()
        consecutive_lost = 0
        MAX_LOST = 15
        print("Spustené. Q = ukončiť.")
        print("Červená bodka = stred čiary")
        print("Zelená čiara = pozícia pásky")
        print("Modrá čiara = stred obrazu")

        try:
            while True:
                frame = self.camera.get_frame()
                if frame is None:
                    continue

                self.frame_count += 1
                error, debug = self.detector.detect(frame)

                if error is not None:
                    consecutive_lost = 0
                    self.last_error = error
                    self.last_seen = time.time()
                    steering = max(-1.0, min(1.0,
                                  self.pid.update(error)))
                    speed = self.compute_speed(error)
                    msg = self.motor.send(speed, steering)
                    fps = (self.frame_count /
                           (time.time() - self.start_time))
                    print(f"\r[{self.frame_count:5d}] "
                          f"err={error:+.3f} "
                          f"steer={steering:+.3f} "
                          f"speed={speed:.2f} "
                          f"cmd={msg} "
                          f"fps={fps:.1f}",
                          end="", flush=True)
                else:
                    consecutive_lost += 1
                    self.lost_count += 1
                    time_lost = time.time() - self.last_seen

                    if time_lost < 0.5:
                        # Krátko stratená — pokračuj posledným smerom
                        steering = self.last_error * 0.5
                        speed = 0.1
                    elif time_lost < 2.0:
                        # Dlhšie stratená — hľadaj otáčaním
                        steering = (1.0 if self.last_error > 0
                                    else -1.0)
                        speed = 0.05
                    else:
                        # Príliš dlho — zastav
                        steering = 0.0
                        speed = 0.0
                        self.motor.stop()
                        self.pid.reset()

                    self.motor.send(speed, steering)
                    print(f"\rČiara stratená {consecutive_lost}x",
                          end="", flush=True)

                if self.show_preview:
                    cv2.imshow("Line Follower", debug)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            print("\nUkončujem...")
        finally:
            self.motor.stop()
            self.motor.cleanup()
            self.camera.stop()
            cv2.destroyAllWindows()
            elapsed = time.time() - self.start_time
            print(f"\nHotovo: {self.frame_count} snímok "
                  f"za {elapsed:.1f}s | "
                  f"stratená: {self.lost_count}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", action="store_true",
                        help="Pouzi serial port pre motor")
    parser.add_argument("--port", type=str,
                        default="/dev/ttyUSB0",
                        help="Serial port")
    parser.add_argument("--no-preview", action="store_true",
                        help="Vypni graficke okno")
    parser.add_argument("--flip", type=int, default=0,
                        choices=[0, 1, 2, 3, 4, 5, 6, 7],
                        help="Otocenie obrazu (0=none, 2=180)")
    args = parser.parse_args()

    Config.FLIP_METHOD = args.flip

    LineFollower(
        show_preview=not args.no_preview,
        use_serial=args.serial,
        serial_port=args.port
    ).run()