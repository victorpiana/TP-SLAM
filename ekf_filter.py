import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, x_init, P_init, Q, R, f, h, jacobian_f, jacobian_h, jacobian_w=None, jacobian_v=None):
        # état initial
        self.x = x_init
        self.P = P_init

        # bruit
        self.Q = Q
        self.R = R

        # fonctions du système
        self.f = f  # prédiction de l'état
        self.h = h  # fonction de mesure
        self.jacobian_f = jacobian_f  # dérivée de f par rapport à x
        self.jacobian_h = jacobian_h  # dérivée de h par rapport à x
        self.jacobian_w = jacobian_w  # bruit de f
        self.jacobian_v = jacobian_v  # bruit de h

    def predict(self, u=None):
        # prédire le prochain état
        self.x = self.f(self.x, u) # x′ = f(xₖ₋₁)
        A = self.jacobian_f(self.x, u)
        W = self.jacobian_w(self.x, u) if self.jacobian_w else np.eye(len(self.x))

        self.P = A @ self.P @ A.T + W @ self.Q @ W.T    # P′ = A P Aᵀ + W Q Wᵀ

    def update(self, z):
        y = z - self.h(self.x)                           # y = xₖ - h(x′)
        H = self.jacobian_h(self.x)
        V = self.jacobian_v(self.x) if self.jacobian_v else np.eye(len(z))
        S = H @ self.P @ H.T + V @ self.R @ V.T          # S = H P′ Hᵀ + V R Vᵀ
        K = self.P @ H.T @ np.linalg.inv(S)              # K = P′ Hᵀ S⁻¹

        self.x = self.x + K @ y                          # x = x′ + K y
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P  # P = (I - K H) P′
