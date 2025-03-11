from flask import Flask, render_template, Response, jsonify
import cv2
import os
import mediapipe as mp
import numpy as np

app = Flask(__name__)


face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
drawing = mp.solutions.drawing_utils

drawing_style = drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
left_eye_style = drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
right_eye_style = drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
lips_style = drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
nose_style = drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
eyebrows_style = drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)

camera_active = True
black_screen = False
show_mesh = True
show_distances = False

eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_pupils(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    eyes = eye_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

    centers = []
    nose_x = None
    nose_y_end = None

    for (x, y, w, h) in faces:
        nose_x = x + w // 2
        nose_y_start = y + h // 4
        nose_y_end = y + (2 * h // 3)

    for (x, y, w, h) in eyes:
        center = (x + w // 2, y + h // 2)
        centers.append(center)
        cv2.circle(frame, center, 5, (255, 255, 255), -1)

    if len(centers) >= 2 and nose_x is not None and nose_y_end is not None:
        centers = sorted(centers, key=lambda c: c[0])
        cv2.line(frame, centers[0], centers[1], (255, 255, 255), 2)
        cv2.line(frame, (nose_x, nose_y_start), (nose_x, nose_y_end), (0, 0, 255), 2)

        pupil_distance = np.linalg.norm(np.array(centers[0]) - np.array(centers[1]))
        left_nose_distance = abs(centers[0][0] - nose_x)
        right_nose_distance = abs(centers[1][0] - nose_x)

        pixel_to_cm_ratio = 6.3 / pupil_distance
        cm_pupil_distance = pupil_distance * pixel_to_cm_ratio
        cm_left_nose_distance = left_nose_distance * pixel_to_cm_ratio
        cm_right_nose_distance = right_nose_distance * pixel_to_cm_ratio

        cv2.line(frame, centers[0], (nose_x, centers[0][1]), (255, 0, 0), 2)
        cv2.line(frame, centers[1], (nose_x, centers[1][1]), (0, 0, 255), 2)

        cv2.putText(frame, f"Pupils: {cm_pupil_distance:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Left-Nose: {cm_left_nose_distance:.2f} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Right-Nose: {cm_right_nose_distance:.2f} cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        if not camera_active:
            continue

        ret, frame = cap.read()
        if not ret:
            break

        if black_screen:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(color)

            if result.multi_face_landmarks and show_mesh:
                for face in result.multi_face_landmarks:
                    drawing.draw_landmarks(frame, face, mp.solutions.face_mesh.FACEMESH_TESSELATION, None, drawing_style)
                    drawing.draw_landmarks(frame, face, mp.solutions.face_mesh.FACEMESH_CONTOURS, None, drawing_style)
                    drawing.draw_landmarks(frame, face, mp.solutions.face_mesh.FACEMESH_LEFT_EYE, None, left_eye_style)
                    drawing.draw_landmarks(frame, face, mp.solutions.face_mesh.FACEMESH_RIGHT_EYE, None, right_eye_style)
                    drawing.draw_landmarks(frame, face, mp.solutions.face_mesh.FACEMESH_LIPS, None, lips_style)
                    drawing.draw_landmarks(frame, face, mp.solutions.face_mesh.FACEMESH_NOSE, None, nose_style)
                    drawing.draw_landmarks(frame, face, mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW, None, eyebrows_style)
                    drawing.draw_landmarks(frame, face, mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW, None, eyebrows_style)

            if show_distances:
                frame = detect_pupils(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_camera')
def toggle_camera():
    global camera_active
    camera_active = not camera_active
    state = "on" if camera_active else "off"
    return jsonify(message=f"Camera {state}")

@app.route('/toggle_screen')
def toggle_screen():
    global black_screen
    black_screen = not black_screen
    state = "black" if black_screen else "normal"
    return jsonify(message=f"Screen {state}")

@app.route('/toggle_mesh')
def toggle_mesh():
    global show_mesh
    show_mesh = not show_mesh
    state = "visible" if show_mesh else "hidden"
    return jsonify(message=f"Mesh {state}")

@app.route('/toggle_distances')
def toggle_distances():
    global show_distances
    show_distances = not show_distances
    state = "enabled" if show_distances else "disabled"
    return jsonify(message=f"Distances {state}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, ssl_context=('/etc/letsencrypt/live/tu-dominio.com/fullchain.pem', '/etc/letsencrypt/live/tu-dominio.com/privkey.pem'))
