import numpy as np
import dlib
import cv2
import pygame
import winsound
import datetime
import time
import requests
from twilio.rest import Client
import threading
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.uix.label import Label

class DrowsinessApp(App):
    stop_flag = threading.Event()  # Declare stop_flag as a class variable
    def build(self):
        self.layout = BoxLayout(orientation='vertical', spacing=10)
        self.start_button = Button(text='Start', on_press=self.start_detection)
        self.stop_button = Button(text='Stop', on_press=self.stop_detection)
        self.status_label = Label(text='Status: Stopped')
        self.layout.add_widget(self.status_label)
        self.layout.add_widget(self.start_button)
        self.layout.add_widget(self.stop_button)
        return self.layout

    def start_detection(self, instance):
        self.status_label.text = 'Status: Running'
        self.stop_flag.clear()  # Clear the stop flag
        threading.Thread(target=self.run_detection).start()

    def stop_detection(self, instance):
        self.status_label.text = 'Status: Stopped'
        self.stop_flag.set()  # Set the stop flag to signal the thread to stop

    def run_detection(self):
        twilio_account_sid = 'AC4c04be22ef2e09f6d0801d3022b96de2'
        twilio_auth_token = '7a028ea780b65c9effee92bcef0dcb24'
        twilio_phone_number = 'whatsapp:+14155238886'
        user_phone_number = 'whatsapp:+919205675535'

        client = Client(twilio_account_sid, twilio_auth_token)

        def upload_video_to_file_io(video_filename):
            try:
                # Upload the video file to file.io
                with open(video_filename, 'rb') as video_file:
                    response = requests.post('https://file.io/', files={'file': video_file})
                    response_data = response.json()
                    print(response_data)
                    return response_data.get('link', '')

            except Exception as e:
                print(f"Failed to upload video to file.io: {str(e)}")
                return None

        def get_longitude_latitude():
            try:
                # Send a GET request to ipinfo.io to retrieve location information
                response = requests.get("https://ipinfo.io")
                data = response.json()

                # Extract latitude and longitude
                latitude, longitude = data.get("loc", "").split(",")

                return float(latitude), float(longitude)

            except Exception as e:
                print(f"Failed to retrieve latitude and longitude: {str(e)}")
                return None, None

        def send_whatsapp_message(video_filename):
            try:
                video_url = upload_video_to_file_io(video_filename)
                print(video_url)
                latitude, longitude = get_longitude_latitude()
                print(latitude)
                print(longitude)
                message = client.messages.create(
                    body=f"LOCATION : https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"
                         f" VIDEO : {video_url}",
                    from_=twilio_phone_number,
                    to=user_phone_number
                )

            except Exception as e:
                print(f"Failed to send WhatsApp message: {str(e)}")

        def calc_aspect_ratio(eye_coords):
            a = np.linalg.norm(eye_coords[1] - eye_coords[5])
            b = np.linalg.norm(eye_coords[2] - eye_coords[4])
            c = np.linalg.norm(eye_coords[0] - eye_coords[3])
            ratio = (a + b) / (2.0 * c)
            return ratio

        alert_threshold = 0.23
        frame_check_interval = 20
        continuous_drowsy_threshold = 1

        face_detector = dlib.get_frontal_face_detector()
        landmark_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

        (left_eye_start, left_eye_end) = (36, 42)
        (right_eye_start, right_eye_end) = (42, 48)

        pygame.mixer.init()
        pygame.mixer.music.load("mysong.mp3")

        video_capture = cv2.VideoCapture(0)

        consecutive_frames = 0
        last_alert_time = None
        music_playing = False

        video_writer = None
        video_start_time = None

        while not self.stop_flag.is_set():  # Check the stop flag in the loop condition
            ret, frame = video_capture.read()
            frame = cv2.resize(frame, (450, 300))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detected_faces = face_detector(gray, 0)
            drowsy_state = False

            for face in detected_faces:
                face_shape = landmark_predictor(gray, face)
                shape_coords = np.array([(face_shape.part(i).x, face_shape.part(i).y) for i in range(68)])

                left_eye_coords = shape_coords[left_eye_start:left_eye_end]
                right_eye_coords = shape_coords[right_eye_start:right_eye_end]

                left_eye_ratio = calc_aspect_ratio(left_eye_coords)
                right_eye_ratio = calc_aspect_ratio(right_eye_coords)

                avg_eye_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

                leye_contour = cv2.convexHull(left_eye_coords)
                reye_contour = cv2.convexHull(right_eye_coords)

                cv2.drawContours(frame, [leye_contour], -1, (255, 0, 0), 1)
                cv2.drawContours(frame, [reye_contour], -1, (255, 0, 0), 1)

                if avg_eye_ratio < alert_threshold:
                    consecutive_frames += 1

                    if consecutive_frames >= frame_check_interval:
                        if last_alert_time is None:
                            last_alert_time = datetime.datetime.now()
                        elapsed_time = (datetime.datetime.now() - last_alert_time).total_seconds()

                        if elapsed_time >= continuous_drowsy_threshold:
                            drowsy_state = True
                else:
                    consecutive_frames = 0
                    last_alert_time = None

            if drowsy_state and not music_playing:
                pygame.mixer.music.play()
                music_playing = True

                video_filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".mp4"
                video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20,
                                               (frame.shape[1], frame.shape[0]))
                video_start_time = datetime.datetime.now()

            if drowsy_state:
                alert_text = "Drowsiness Detected!"
                alert_color = (0, 0, 255)  # Red color for alert

                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), alert_color, -1)
                cv2.putText(frame, alert_text, (10, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                winsound.Beep(1000, 500)

                if not music_playing:
                    pygame.mixer.music.play()
                    music_playing = True

                if video_writer is None:
                    video_filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".mp4"
                    video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20,
                                                   (frame.shape[1], frame.shape[0]))
                    video_start_time = datetime.datetime.now()

                # Continue recording for 5 seconds after drowsiness is detected
            if video_writer is not None and (datetime.datetime.now() - video_start_time).total_seconds() <= 10:
                video_writer.write(frame)
            else:
                if video_writer is not None:
                    video_writer.release()
                    send_whatsapp_message(video_filename)
                    print("WhatsApp message sent")

                video_writer = None

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        video_capture.release()

if __name__ == '__main__':
        DrowsinessApp().run()

