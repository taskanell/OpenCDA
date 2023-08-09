import numpy as np


class SpeedKalmanFilter(object):

    def __init__(self, dt):
        # Define the state transition matrix (A) and measurement matrix (H)
        self.time_step = dt
        self.A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        # Define the process noise covariance matrix (Q) and measurement noise covariance matrix (R)
        sigma_v = 0.1  # standard deviation of velocity
        sigma_x = 0.1  # standard deviation of position
        sigma_y = 0.1
        self.Q = np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                      [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                      [dt ** 3 / 2, 0, dt ** 2, 0],
                      [0, dt ** 3 / 2, 0, dt ** 2]]) * sigma_v ** 2
        self.R = np.array([[sigma_x ** 2, 0],
                      [0, sigma_y ** 2]])

        self.xEst = np.zeros((4, 1))
        self.PEst = np.eye(4)

    def run_step_init(self, x, y, xSpeed, ySpeed):
        """
        Initalization for states.

        Args:
            -x (float): The X coordinate.
            -y (float): Tehe y coordinate.
            -heading (float): The heading direction.
            -velocity (float): The velocity.

        """
        self.xEst[0] = x
        self.xEst[1] = y
        self.xEst[2] = xSpeed
        self.xEst[3] = ySpeed

    def run_step(self, x, P, z):

        # Predict the state and covariance
        x = self.A.dot(x)
        P = self.A.dot(P).dot(self.A.T) + self.Q

        # Compute the Kalman gain
        K = P.dot(self.H.T).dot(np.linalg.inv(self.H.dot(P).dot(self.H.T) + self.R))

        # Update the state and covariance
        x = x + K.dot(z - self.H.dot(x))
        P = (np.eye(4) - K.dot(self.H)).dot(P)

        # Compute the speed estimate
        speed_x = x[2, 0]
        speed_y = x[3, 0]

        return x, P, (speed_x, speed_y)
