import numpy as np
from numpy import ndarray
import math

def AngleAxisToQuat(angleAxis):
    angle = np.linalg.norm(angleAxis)
    if angle < 1e-8:  # angle trop petit : pas de rotation significative
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = angleAxis / angle
    sinA = np.sin(angle / 2.0)
    cosA = np.cos(angle / 2.0)

    return np.array([
        cosA,
        axis[0] * sinA,
        axis[1] * sinA,
        axis[2] * sinA
    ])

def QuatMultiplicationJacobian(dt, quaternionToMatrix):
    dQNewOndQold = ndarray(shape=(4, 4), dtype=np.float64)
    dQNewOndQold[0][0] = dt * quaternionToMatrix[0]
    dQNewOndQold[0][1] = -dt * quaternionToMatrix[1]
    dQNewOndQold[0][2] = -dt * quaternionToMatrix[2]
    dQNewOndQold[0][3] = -dt * quaternionToMatrix[3]

    dQNewOndQold[1][0] = dt * quaternionToMatrix[1]
    dQNewOndQold[1][1] = dt * quaternionToMatrix[0]
    dQNewOndQold[1][2] = -dt * quaternionToMatrix[3]
    dQNewOndQold[1][3] = dt * quaternionToMatrix[2]

    dQNewOndQold[2][0] = dt * quaternionToMatrix[2]
    dQNewOndQold[2][1] = dt * quaternionToMatrix[3]
    dQNewOndQold[2][2] = dt * quaternionToMatrix[0]
    dQNewOndQold[2][3] = -dt * quaternionToMatrix[1]

    dQNewOndQold[3][0] = dt * quaternionToMatrix[3]
    dQNewOndQold[3][1] = -dt * quaternionToMatrix[2]
    dQNewOndQold[3][2] = dt * quaternionToMatrix[1]
    dQNewOndQold[3][3] = dt * quaternionToMatrix[0]

    return dQNewOndQold

def OpposeQuatMultiplicationJacobian(dt, quaternionToMatrix):
    dQNewOndQold = ndarray(shape=(4, 4), dtype=np.float64)
    dQNewOndQold[0][0] = dt * quaternionToMatrix[0]
    dQNewOndQold[0][1] = -dt * quaternionToMatrix[1]
    dQNewOndQold[0][2] = -dt * quaternionToMatrix[2]
    dQNewOndQold[0][3] = -dt * quaternionToMatrix[3]

    dQNewOndQold[1][0] = dt * quaternionToMatrix[1]
    dQNewOndQold[1][1] = dt * quaternionToMatrix[0]
    dQNewOndQold[1][2] = dt * quaternionToMatrix[3]
    dQNewOndQold[1][3] = -dt * quaternionToMatrix[2]

    dQNewOndQold[2][0] = dt * quaternionToMatrix[2]
    dQNewOndQold[2][1] = -dt * quaternionToMatrix[3]
    dQNewOndQold[2][2] = dt * quaternionToMatrix[0]
    dQNewOndQold[2][3] = dt * quaternionToMatrix[1]

    dQNewOndQold[3][0] = dt * quaternionToMatrix[3]
    dQNewOndQold[3][1] = dt * quaternionToMatrix[2]
    dQNewOndQold[3][2] = -dt * quaternionToMatrix[1]
    dQNewOndQold[3][3] = dt * quaternionToMatrix[0]

    return dQNewOndQold

def DerivateAngularVelocityQuaternionOnAngularVelocity(dt, angleAxis):
    #
    # It will be more complex as we need to mix the conversion from angle axis to quaternion and quaternion multiplication
    angle = np.linalg.norm(angleAxis[0:3])
    jacobianAngleAxisToQuat = ndarray(shape = (4,3), dtype = np.float64)

    #
    # Partial derivate for w for all component
    jacobianAngleAxisToQuat[0][0] = DerivateWComponentOnAngularVelocity(dt, angleAxis[0], angle)
    jacobianAngleAxisToQuat[0][1] = DerivateWComponentOnAngularVelocity(dt, angleAxis[1], angle)
    jacobianAngleAxisToQuat[0][2] = DerivateWComponentOnAngularVelocity(dt, angleAxis[2], angle)

    #
    # Partial derivate for x for all component
    jacobianAngleAxisToQuat[1][0] = DerivateAComponentOnA(dt, angleAxis[0], angle)
    jacobianAngleAxisToQuat[1][1] = DerivateAComponentOnB(dt, angleAxis[0], angleAxis[1], angle)
    jacobianAngleAxisToQuat[1][2] = DerivateAComponentOnB(dt, angleAxis[0], angleAxis[2], angle)

    #
    # Partial derivate for y for all component
    jacobianAngleAxisToQuat[2][0] = DerivateAComponentOnB(dt, angleAxis[1], angleAxis[0], angle)
    jacobianAngleAxisToQuat[2][1] = DerivateAComponentOnA(dt, angleAxis[1], angle)
    jacobianAngleAxisToQuat[2][2] = DerivateAComponentOnB(dt, angleAxis[0], angleAxis[2], angle)

    #
    # Partial derivate for z for all component
    jacobianAngleAxisToQuat[3][0] = DerivateAComponentOnB(dt, angleAxis[2], angleAxis[0], angle)
    jacobianAngleAxisToQuat[3][1] = DerivateAComponentOnB(dt, angleAxis[2], angleAxis[1], angle)
    jacobianAngleAxisToQuat[3][2] = DerivateAComponentOnA(dt, angleAxis[2], angle)

    return jacobianAngleAxisToQuat

def DerivateWComponentOnAngularVelocity(dt, value, angle):
    #
    # The formula of the conversion from axis angle for w component of quaternion
    # w = cos(angle / 2.0)
    # with angle = sqrt(x*x + y*y + z*z)
    # By applying the partial derivative (computed by wxmaxima), we get the result below
    if angle < 1e-8:
        return 0.0
    return -dt / 2. * value / angle * math.sin(dt * angle / 2.)


def DerivateAComponentOnA(dt, value, angle):
    if angle < 1e-8:
        return 0.0
    return dt/2. * value * value * math.cos(dt * angle / 2.) / (angle * angle) + \
        (1.0 / angle - value * value / (angle * angle * angle)) * math.sin(dt * angle / 2.)

def DerivateAComponentOnB(dt, valueA, valueB, angle):
    if angle < 1e-8:
        return 0.0
    return dt / 2.0 * valueA * valueB / (angle * angle) * math.cos(dt * angle / 2.) - \
        valueA * valueB / (angle * angle * angle) * math.sin(dt * angle / 2.)


def DRenormQiOnQi(value, norm):
    #
    # The formula of the renormalization of each component is described as this :
    # qi = i / sqrt(w*w + x*x + y*y + z*z)
    # By applying the partial derivative (computed by wxmaxima), we get the result below
    return (1.0 - value * value / (norm * norm)) * norm

def DRenormQiOnQj(valueA, valueB, norm):
    #
    # The formula of the renormalization of each component is described as this :
    # qj = i / sqrt(w*w + x*x + y*y + z*z)
    # with j which can be any component but j != i
    # By applying the partial derivative (computed by wxmaxima), we get the result below
    return -valueA * valueB / (norm * norm * norm)

def RenormalizationJacobian(state: np.ndarray):
    renoramlizationJacobian = np.ndarray(shape=(4, 4), dtype = np.float64)

    #
    # Compute the norm of the quaternion
    norm = np.linalg.norm(state[6:10])

    #
    # Compute the partial derivative for each component
    # For w component
    renoramlizationJacobian[0][0] = DRenormQiOnQi(state[6], norm)
    renoramlizationJacobian[0][1] = DRenormQiOnQj(state[6], state[7], norm)
    renoramlizationJacobian[0][2] = DRenormQiOnQj(state[6], state[8], norm)
    renoramlizationJacobian[0][3] = DRenormQiOnQj(state[6], state[9], norm)

    # For x component
    renoramlizationJacobian[1][0] = DRenormQiOnQj(state[7], state[6], norm)
    renoramlizationJacobian[1][1] = DRenormQiOnQi(state[7], norm)
    renoramlizationJacobian[1][2] = DRenormQiOnQj(state[7], state[8], norm)
    renoramlizationJacobian[1][3] = DRenormQiOnQj(state[7], state[9], norm)

    # For y component
    renoramlizationJacobian[2][0] = DRenormQiOnQj(state[8], state[6], norm)
    renoramlizationJacobian[2][1] = DRenormQiOnQj(state[8], state[7], norm)
    renoramlizationJacobian[2][2] = DRenormQiOnQi(state[8], norm)
    renoramlizationJacobian[2][3] = DRenormQiOnQj(state[8], state[9], norm)

    # For z component
    renoramlizationJacobian[3][0] = DRenormQiOnQj(state[9], state[6], norm)
    renoramlizationJacobian[3][1] = DRenormQiOnQj(state[9], state[7], norm)
    renoramlizationJacobian[3][2] = DRenormQiOnQj(state[9], state[8], norm)
    renoramlizationJacobian[3][3] = DRenormQiOnQi(state[9], norm)

    # return the result
    return renoramlizationJacobian

def PredictionJacobian(dt, state: ndarray):
    # The position/velocity is pretty trivial to compute as the computation is linear
    posVeloJacobian = np.identity(6, dtype=np.float64)
    posVeloJacobian[0][3] = dt
    posVeloJacobian[1][4] = dt
    posVeloJacobian[2][5] = dt

    # Construct part of partial derivative for orientation
    DQNewOnQOld = OpposeQuatMultiplicationJacobian(dt, state[6:10])

    #
    # Part of the angular velocity
    angularQuat = AngleAxisToQuat(state[10:13])
    DQNewOnQOmega = QuatMultiplicationJacobian(dt, angularQuat)
    DQNewOnDOmega = DerivateAngularVelocityQuaternionOnAngularVelocity(dt, state[10:13])
    Renormalization = RenormalizationJacobian(state)
    DQNewOnDOmega = Renormalization @ DQNewOnQOmega @ DQNewOnDOmega

    # Construct the final jacobian
    predictionJacobian = np.identity(state.shape[0], dtype=np.float64)

    #
    # We know that the position velocity is linear so the partial derivate will be the linear coefficient.

    # position on velocity
    predictionJacobian[0][3] = dt
    predictionJacobian[1][4] = dt
    predictionJacobian[2][5] = dt

    # orientation (dQnewOndQOmega)
    predictionJacobian[np.ix_([6, 7, 8, 9], [6, 7, 8, 9])] = DQNewOnQOld

    # angular velocity (dQnewOnDOmega)
    predictionJacobian[np.ix_([6, 7, 8, 9], [10, 11, 12])] = DQNewOnDOmega
    return predictionJacobian

def PredictionNoiseJacobian(dt, state: ndarray):
    # The construction can be found with the equation of the prediction https://www.researchgate.net/publication/6397818_MonoSLAM_real-time_single_camera_SLAM
    # As we see the noise impacts only the velocity in the position, so I * dt
    # However, the noise impacts the angular velocity in the orientation, so we will compute again the jacobian (which are identical as prediction jacobian for angular velocity)
    # In this implementation of EKF-SLAM, we need to compute the noise integrally so Qw * W * Qw^T
    covarianceNoisePrediction = np.zeros(shape=(6, 6), dtype=np.float64)

    #
    # Set only noise covariance for robot state
    # (Comes from https://github.com/hanmekim/SceneLib2/blob/master/scenelib2/motion_model.cpp)
    covarianceNoisePrediction[0][0]     = 4. * 4. * dt * dt
    covarianceNoisePrediction[1][1]     = 4. * 4. * dt * dt
    covarianceNoisePrediction[2][2]     = 4. * 4. * dt * dt
    covarianceNoisePrediction[3][3]     = 6. * 6. * dt * dt
    covarianceNoisePrediction[4][4]     = 6. * 6. * dt * dt
    covarianceNoisePrediction[5][5]     = 6. * 6. * dt * dt

    #
    # Jacobian of the noise on the computation of the next state
    # Only the velocity and the angular velocity
    noiseJacobian = np.zeros(shape=(13, 6), dtype=np.float64)

    #
    # Compute identity for reuse
    identity = np.identity(3, dtype=np.float64)

    # identity for the velocity and angular velocity as the state stay the same for the next step
    noiseJacobian[np.ix_([3, 4, 5], [0, 1, 2])] = identity
    noiseJacobian[np.ix_([10, 11, 12], [3, 4, 5])] = identity

    #
    # identity * dt for the position for the noise of the velocity as described in Mono SLAM paper
    noiseJacobian[np.ix_([0, 1, 2], [0, 1, 2])] = identity * dt

    #
    # Compute the jacobian for the orientation. Here we only need the dQnewOndOmega
    angularQuat = AngleAxisToQuat(state[10:13])
    DQNewOnQOmega = QuatMultiplicationJacobian(dt, angularQuat)
    DQNewOnDOmega = DerivateAngularVelocityQuaternionOnAngularVelocity(dt, state[10:13])
    DQNewOnDOmega = DQNewOnQOmega @ DQNewOnDOmega

    noiseJacobian[np.ix_([6, 7, 8, 9], [3, 4, 5])] = DQNewOnDOmega

    subPartNoise = noiseJacobian @ covarianceNoisePrediction @ np.transpose(noiseJacobian)
    return subPartNoise
