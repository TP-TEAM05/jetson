#!/usr/bin/env python3
import cv2
import numpy as np
import time
import argparse
import serial
from pyvesc import encode
from pyvesc.VESC.messages import SetRPM, SetServoPosition

# python -m pip install https://github.com/LiamBindle/PyVESC/archive/master.zip
# pyvesc ma neoficialny fix takto ho importuj

class Config:
    # Camera Settings
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 60
    FLIP_METHOD = 0
    
    # Vision ROI & Thresholds
    ROI_START = 0.55
    ROI_END = 0.95
    MIN_CONTOUR_AREA = 300
    MAX_CONTOUR_AREA = 80000
    
    # PID Constants
    KP = 0.40
    KI = 0.001
    KD = 0.08
    
    # VESC Speed Settings (ERPM)
    # ERPM = Speed (m/s) * (Pole Pairs * Gear Ratio * 60) / (Wheel Circumference)
    BASE_SPEED_ERPM = 3500  
    MAX_SPEED_ERPM = 6000
    MIN_SPEED_ERPM = 1500
    
    # VESC Steering Settings (0.0 to 1.0)
    # Adjust these based on your specific F1TENTH servo mechanical limits
    SERVO_CENTER = 0.500
    SERVO_MAX_RIGHT = 0.150  # Usually lower value = right
    SERVO_MAX_LEFT = 0.850   # Usually higher value = left
    SERVO_GAIN = 0.350       # Multiplier for normalized error (-1.0 to 1.0)


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
        self.integral = max(-1.0, min(1.0, self.integral + error * dt))
        derivative = (error - self.prev_error) / dt
        output = (self.kp * error + self.ki * self.integral + self.kd * derivative)
        self.prev_error = error
        self.prev_time = now
        return output


class VESCController:
    def __init__(self, use_serial=True, serial_port="/dev/ttyACM0", debug=False):
        self.use_serial = use_serial and not debug
        self.debug = debug
        self.ser = None
        if self.use_serial:
            try:
                # VESC usually runs at 115200 baud
                self.ser = serial.Serial(serial_port, 115200, timeout=0.05)
                print(f"VESC MK6 connected on: {serial_port}")
            except Exception as e:
                print(f"CRITICAL: VESC Unavailable: {e}")
                self.use_serial = False
        if self.debug:
            print("[DEBUG MODE] VESC output will be printed, not sent.")

    def send(self, erpm, steering_normalized):
        """
        erpm: Speed in Electrical RPM
        steering_normalized: -1.0 (Left) to 1.0 (Right)
        """
        # Map normalized steering (-1.0 to 1.0) to VESC Servo Range (0.0 to 1.0)
        # Note: Invert steering_normalized if left/right are backwards on your chassis
        servo_val = Config.SERVO_CENTER + (steering_normalized * Config.SERVO_GAIN)
        servo_val = max(Config.SERVO_MAX_RIGHT, min(Config.SERVO_MAX_LEFT, servo_val))

        if self.debug:
            print(f"[DEBUG] Would send: ERPM={int(erpm)}, Servo={servo_val:.3f}")
        elif self.use_serial and self.ser:
            try:
                # Encode and send speed
                self.ser.write(encode(SetRPM(int(erpm))))
                # Encode and send steering
                self.ser.write(encode(SetServoPosition(float(servo_val))))
            except Exception as e:
                print(f"VESC Write Error: {e}")
        return f"ERPM: {int(erpm)}, Srv: {servo_val:.3f}"

    def stop(self):
        if self.debug:
            print("[DEBUG] Would send: STOP (ERPM=0, Servo=CENTER)")
        elif self.use_serial and self.ser:
            self.ser.write(encode(SetRPM(0)))
            self.ser.write(encode(SetServoPosition(Config.SERVO_CENTER)))

    def cleanup(self):
        self.stop()
        if self.ser:
            self.ser.close()


class Camera:
    def __init__(self, use_usb=False, usb_index=0):
        self.cap = None
        self.use_usb = use_usb
        self.usb_index = usb_index

    def start(self):
        if self.use_usb:
            self.cap = cv2.VideoCapture(self.usb_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open USB Camera at index {self.usb_index}.")
            print(f"USB Camera Initialized: {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT}")
        else:
            # Optimized GStreamer pipeline for Jetson Orin
            pipeline = (
                f"nvarguscamerasrc ! "
                f"video/x-raw(memory:NVMM), width={Config.FRAME_WIDTH}, height={Config.FRAME_HEIGHT}, "
                f"format=NV12, framerate={Config.FPS}/1 ! "
                f"nvvidconv flip-method={Config.FLIP_METHOD} ! "
                f"video/x-raw, format=BGRx ! "
                f"videoconvert ! video/x-raw, format=BGR ! "
                f"appsink drop=1 max-buffers=1"
            )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open CSI Camera.")
            print(f"CSI Camera Initialized: {Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT} @ {Config.FPS}fps")

    def get_frame(self):
        if self.cap is None: return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def stop(self):
        if self.cap: self.cap.release()

class GPU_LineDetector:
    def __init__(self):
        # Gracefully check if CUDA is available to prevent crashes on standard PCs
        self.use_cuda = False
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.use_cuda = True
                self.gpu_frame = cv2.cuda_GpuMat()
                self.gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (11, 11), 0)
        except AttributeError:
            pass # cv2.cuda module doesn't exist in this OpenCV build
            
        if self.use_cuda:
            print("[INFO] CUDA Detected! Running accelerated vision pipeline.")
        else:
            print("[WARNING] CUDA not detected. Falling back to CPU for Line Detection.")

    def detect(self, frame):
        h, w = frame.shape[:2]
        y_start = int(h * Config.ROI_START)
        y_end = int(h * Config.ROI_END)
        
        # Optimization: Crop ROI first to save memory/bandwidth
        roi = frame[y_start:y_end, :]
        
        if self.use_cuda:
            # --- GPU Pipeline (Jetson Orin) ---
            self.gpu_frame.upload(roi)
            gpu_gray = cv2.cuda.cvtColor(self.gpu_frame, cv2.COLOR_BGR2GRAY)
            gpu_blurred = self.gaussian_filter.apply(gpu_gray)
            
            # NOTE: cv2.cuda.threshold DOES NOT support Otsu! 
            # We must download the blurred image and do Otsu on the CPU.
            blurred = gpu_blurred.download()
        else:
            # --- CPU Pipeline (Windows Laptop Testing) ---
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        # Otsu's Thresholding (Executes on CPU)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = np.ones((7, 7), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        debug_img = frame.copy()
        
        if not contours:
            return None, debug_img

        valid_contours = [c for c in contours if Config.MIN_CONTOUR_AREA < cv2.contourArea(c) < Config.MAX_CONTOUR_AREA]
        if not valid_contours:
            return None, debug_img

        largest = max(valid_contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0: return None, debug_img
        
        cx = int(M["m10"] / M["m00"])
        
        # Calculate normalized error (-1.0 to 1.0)
        frame_center = w // 2
        error = (cx - frame_center) / frame_center
        
        # Draw debug info
        cv2.circle(debug_img, (cx, int(M["m01"] / M["m00"]) + y_start), 10, (0, 0, 255), -1)
        cv2.line(debug_img, (frame_center, y_start), (frame_center, y_end), (255, 0, 0), 2)
        cv2.rectangle(debug_img, (0, y_start), (w, y_end), (255, 255, 0), 2)
        
        return error, debug_img

class F1TenthRacer:
    def __init__(self, show_preview=True, use_serial=True, serial_port="/dev/ttyACM0", debug=False, use_usb_cam=False, usb_index=0):
        self.show_preview = show_preview
        self.pid = PID(Config.KP, Config.KI, Config.KD)
        self.detector = GPU_LineDetector()
        self.motor = VESCController(use_serial, serial_port, debug=debug)
        self.camera = Camera(use_usb=use_usb_cam, usb_index=usb_index)
        self.frame_count = 0
        self.start_time = time.time()
        self.last_error = 0.0

    def compute_erpm(self, error):
        # Slow down in corners based on the error magnitude
        erpm = Config.BASE_SPEED_ERPM * (1.0 - 0.4 * abs(error))
        return max(Config.MIN_SPEED_ERPM, min(Config.MAX_SPEED_ERPM, erpm))

    def run(self):
        self.camera.start()
        print("F1TENTH GPU Pipeline Active. Press 'q' to quit.")

        try:
            while True:
                frame = self.camera.get_frame()
                if frame is None: continue
                
                self.frame_count += 1
                error, debug = self.detector.detect(frame)

                if error is not None:
                    self.last_error = error
                    
                    # Compute PID steering and limit to [-1.0, 1.0]
                    steering_normalized = max(-1.0, min(1.0, self.pid.update(error)))
                    
                    # Compute dynamic speed
                    erpm = self.compute_erpm(error)
                    
                    # Dispatch to VESC
                    msg = self.motor.send(erpm, steering_normalized)
                    
                    fps = self.frame_count / (time.time() - self.start_time)
                    print(f"\r[{self.frame_count:5d}] err={error:+.2f} PID={steering_normalized:+.2f} {msg} fps={fps:.1f}", end="", flush=True)
                else:
                    # Line lost - Emergency Brake
                    self.motor.stop()
                    self.pid.reset()
                    print("\r[WARNING] Line Lost! Braking...", end="", flush=True)

                if self.show_preview:
                    cv2.imshow("F1TENTH Vision", debug)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            print("\nRace Interrupted by User.")
        finally:
            self.motor.cleanup()
            self.camera.stop()
            cv2.destroyAllWindows()
            print("\nShutdown Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", action="store_true", help="Enable VESC Serial output")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="VESC Serial port (usually /dev/ttyACM0 or /dev/ttyUSB0)")
    parser.add_argument("--no-preview", action="store_true", help="Disable OpenCV imshow (increases FPS)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (no VESC, print commands)")
    parser.add_argument("--usb-cam", action="store_true", help="Use USB camera instead of CSI camera")
    parser.add_argument("--usb-index", type=int, default=0, help="USB camera index (default 0)")
    args = parser.parse_args()

    F1TenthRacer(
        show_preview=not args.no_preview,
        use_serial=args.serial,
        serial_port=args.port,
        debug=args.debug,
        use_usb_cam=args.usb_cam,
        usb_index=args.usb_index
    ).run()