import cv2
import numpy as np
from ekf_filter import ExtendedKalmanFilter  # on importe la classe EKF depuis le fichier séparé
import matplotlib.pyplot as plt

# === Partie matplotlib réservée à l'affichage de la courbe pour l'exercice 3 (pas demandé mais fait en plus) ===
plt.ion()
fig, ax = plt.subplots()
z_brutes = []
z_filtres = []
line_brute, = ax.plot([], [], 'r--', label='Z brute')
line_filtrée, = ax.plot([], [], 'g-', label='Z filtrée')
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.set_title("Position Z - brute vs filtrée")
ax.set_xlabel("Frame")
ax.set_ylabel("Z (profondeur)")
ax.legend()

# === paramètres caméra (ici on met des valeurs fictives) ===
# normalement ça vient d'une vraie calibration
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.zeros((5, 1))  # on suppose qu’il n’y a pas de distorsion de l’image

# === configuration ArUco ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()  # paramètres de détection par défaut

# === temps entre deux frames (dt) ===
# on suppose que la caméra tourne à 30 images/sec
dt = 1.0 / 30.0

# === définition des fonctions pour l’EKF étendu avec angles d’Euler ===

# fonction de prédiction de l’état (on ajoute aussi les angles et leurs vitesses)
def f(x, u=None):
    x_new = x.copy()
    # position = position + vitesse * dt
    x_new[0:3] += x[3:6] * dt
    # angles = angles + vitesses angulaires * dt
    x_new[6:9] += x[9:12] * dt
    return x_new

# jacobienne de la fonction f par rapport à x
def jacobian_f(x, u=None):
    A = np.eye(12)
    A[0, 3] = dt
    A[1, 4] = dt
    A[2, 5] = dt
    A[6, 9] = dt
    A[7, 10] = dt
    A[8, 11] = dt
    return A

# jacobienne du bruit de prédiction
def jacobian_w(x, u=None):
    return np.eye(12)

# fonction de mesure h(x) -> on récupère position + angles
def h(x):
    return x[[0, 1, 2, 6, 7, 8]]  # on récupère [x, y, z, rx, ry, rz]

# jacobienne de la mesure h(x)
def jacobian_h(x):
    H = np.zeros((6, 12))
    H[0, 0] = 1  # x
    H[1, 1] = 1  # y
    H[2, 2] = 1  # z
    H[3, 6] = 1  # angle x
    H[4, 7] = 1  # angle y
    H[5, 8] = 1  # angle z
    return H

# jacobienne du bruit de mesure
def jacobian_v(x):
    return np.eye(6)

# === initialisation du filtre EKF avec position + vitesse + angles + vitesses angulaires ===
x_init = np.zeros(12)  # on part avec un état nul
P_init = np.eye(12) * 0.1  # incertitude modérée
Q = np.eye(12) * 0.01  # bruit modèle
R = np.eye(6) * 0.05   # bruit sur mesure : position + angles

# on crée l'objet EKF
ekf = ExtendedKalmanFilter(x_init, P_init, Q, R, f, h, jacobian_f, jacobian_h, jacobian_w, jacobian_v)

# === ouverture de la webcam ===
cap = cv2.VideoCapture(0)

# === boucle principale ===
while True:
    ret, frame = cap.read()
    if not ret:
        break  # si la caméra renvoie rien, on sort

    # détection des marqueurs ArUco dans l’image
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        # estimation de la pose du marqueur
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            # affichage du repère 3D sur l’image (axes X, Y, Z)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # récupération position brute
            position = tvecs[i].ravel()

            # conversion de rvec en matrice de rotation
            R_mat, _ = cv2.Rodrigues(rvecs[i])

            # calcul des angles d’Euler à partir de la matrice (en radians)
            sy = np.sqrt(R_mat[0, 0] ** 2 + R_mat[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                x_angle = np.arctan2(R_mat[2, 1], R_mat[2, 2])
                y_angle = np.arctan2(-R_mat[2, 0], sy)
                z_angle = np.arctan2(R_mat[1, 0], R_mat[0, 0])
            else:
                x_angle = np.arctan2(-R_mat[1, 2], R_mat[1, 1])
                y_angle = np.arctan2(-R_mat[2, 0], sy)
                z_angle = 0

            angles = np.array([x_angle, y_angle, z_angle])

            # on crée le vecteur de mesure [position + angles]
            z = np.concatenate((position, angles))

            # === EKF ===
            ekf.predict()  # prédiction
            ekf.update(z)  # correction avec mesure

            # affichage console
            print("---")
            print("Position brute     :", position)
            print("Angles d'Euler     :", angles)
            print("Position filtrée   :", ekf.x[:3])
            print("Angles filtrés     :", ekf.x[6:9])

            # affichage en live de Z
            z_brutes.append(position[2])
            z_filtres.append(ekf.x[2])

            if len(z_brutes) > 100:
                z_brutes = z_brutes[-100:]
                z_filtres = z_filtres[-100:]

            line_brute.set_data(range(len(z_brutes)), z_brutes)
            line_filtrée.set_data(range(len(z_filtres)), z_filtres)
            ax.set_xlim(0, len(z_brutes))
            ax.set_ylim(min(z_brutes + z_filtres) - 0.01, max(z_brutes + z_filtres) + 0.01)
            plt.pause(0.001)

    # affichage image
    cv2.imshow('EKF Pose ArUco', frame)

    # touche Q pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# fermeture propre
cap.release()
cv2.destroyAllWindows()
