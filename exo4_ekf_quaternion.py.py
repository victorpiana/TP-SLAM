import cv2
import numpy as np
from ekf_filter import ExtendedKalmanFilter
import matplotlib.pyplot as plt
from quaternion_utils import (
    AngleAxisToQuat,
    PredictionJacobian,
    PredictionNoiseJacobian
)
def rvec_to_quaternion(rvec):
    R, _ = cv2.Rodrigues(rvec)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    return np.array([qw, qx, qy, qz])

# === Paramètres caméra ===
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))
dt = 1.0 / 30.0  # 30 FPS

# === Configuration ArUco ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# === Fonction de prédiction EKF ===
def f(x, u=None):
    x_new = x.copy()
    x_new[0:3] += x[3:6] * dt

    q = x[6:10]
    omega = x[10:13]

    

    q_delta = AngleAxisToQuat(omega * dt)

    # q_delta vient de AngleAxisToQuat 
    q_delta = q_delta.flatten()  

    q_new = quaternion_multiply(q, q_delta)

    norm = np.linalg.norm(q_new)
    if norm < 1e-6:
        x_new[6:10] = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        x_new[6:10] = q_new / norm



    return x_new

# === Multiplication quaternion (utile pour f) ===
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

# === Jacobienne de la prédiction ===
def jacobian_f(x, u=None):
    return PredictionJacobian(dt, x)

# === Jacobienne du bruit de la prédiction ===
def jacobian_w(x, u=None):
    return np.eye(13)  # bruit isotrope (W = I)

# === Fonction de mesure : position + quaternion ===
def h(x):
    return np.concatenate([x[0:3], x[6:10]])

# === Jacobienne de h(x) ===
def jacobian_h(x):
    H = np.zeros((7, 13))
    H[0:3, 0:3] = np.eye(3)      # position
    H[3:7, 6:10] = np.eye(4)     # quaternion
    return H

def jacobian_v(x):
    return np.eye(7)

# === Initialisation EKF ===
x_init = np.zeros(13)
x_init[6] = 1.0  # quaternion unité
P_init = np.eye(13) * 0.1
Q = np.eye(13) * 0.01
R = np.eye(7) * 0.05

ekf = ExtendedKalmanFilter(x_init, P_init, Q, R, f, h, jacobian_f, jacobian_h, jacobian_w, jacobian_v)

# === Affichage matplotlib (Z brute vs filtrée) ===
plt.ion()
fig, ax = plt.subplots()
z_brutes, z_filtres = [], []
line_brute, = ax.plot([], [], 'r--', label='Z brute')
line_filtrée, = ax.plot([], [], 'g-', label='Z filtrée')
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.set_title("Z brute vs filtrée")
ax.set_xlabel("Frame")
ax.set_ylabel("Z (profondeur)")
ax.legend()

# === Capture caméra ===
cap = cv2.VideoCapture(0)
frame_idx = 0

last_quat = None  # pour calculer omega
while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

            position = tvecs[i].ravel()
            quat = rvec_to_quaternion(rvecs[i])

            # === Estimation d'omega entre deux frames ===
            if last_quat is not None:
                # q_delta = q_current * q_last^-1
                q1 = quat / np.linalg.norm(quat)
                q0 = last_quat / np.linalg.norm(last_quat)

                # Quaternion inverse
                q0_conj = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
                q_delta = quaternion_multiply(q1, q0_conj)

                # Conversion en rotation vectorielle
                angle = 2 * np.arccos(np.clip(q_delta[0], -1.0, 1.0))
                if angle < 1e-6:
                    omega = np.zeros(3)
                else:
                    axis = q_delta[1:4] / np.sin(angle / 2)
                    omega = (angle / dt) * axis
            else:
                omega = np.zeros(3)

            last_quat = quat

            # Mettre à jour omega dans le vecteur d'état
            ekf.x[10:13] = omega

            # Mesure z = position + quaternion
            z = np.concatenate([position, quat])

            # EKF
            ekf.predict()
            ekf.update(z)
            if frame_idx % 10 == 0:
                print(f"Frame {frame_idx:03d} | Z brute : {position[2]:.3f} | Z filtrée : {ekf.x[2]:.3f}")
                print(f"omega = {omega} | norm = {np.linalg.norm(omega):.4f}")

            # === Courbe matplotlib pour afficher l’évolution de Z ===
            z_brutes.append(position[2])
            z_filtres.append(ekf.x[2])

            # On garde seulement les 100 dernières valeurs pour éviter surcharge mémoire
            if len(z_brutes) > 100:
                z_brutes = z_brutes[-100:]
                z_filtres = z_filtres[-100:]

            # mise à jour des courbes affichées
            line_brute.set_data(range(len(z_brutes)), z_brutes)
            line_filtrée.set_data(range(len(z_filtres)), z_filtres)
            ax.set_xlim(0, len(z_brutes))
            ax.set_ylim(min(z_brutes + z_filtres) - 0.01, max(z_brutes + z_filtres) + 0.01)
            plt.pause(0.001)  # pause pour laisser le temps à matplotlib de mettre à jour
            frame_idx += 1
    cv2.imshow("Aruco Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()