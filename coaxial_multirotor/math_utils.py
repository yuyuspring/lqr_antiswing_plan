import numpy as np


DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


def clamp(value, lower, upper):
    return np.minimum(np.maximum(value, lower), upper)


def skew(vector: np.ndarray) -> np.ndarray:
    x_val, y_val, z_val = vector
    return np.array(
        [
            [0.0, -z_val, y_val],
            [z_val, 0.0, -x_val],
            [-y_val, x_val, 0.0],
        ]
    )


def euler321_to_quaternion(euler_rad: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = euler_rad
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = lhs
    w2, x2, y2, z2 = rhs
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    return quaternion / np.linalg.norm(quaternion)


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w_val, x_val, y_val, z_val = normalize_quaternion(quaternion)
    return np.array(
        [
            [1 - 2 * (y_val**2 + z_val**2), 2 * (x_val * y_val - z_val * w_val), 2 * (x_val * z_val + y_val * w_val)],
            [2 * (x_val * y_val + z_val * w_val), 1 - 2 * (x_val**2 + z_val**2), 2 * (y_val * z_val - x_val * w_val)],
            [2 * (x_val * z_val - y_val * w_val), 2 * (y_val * z_val + x_val * w_val), 1 - 2 * (x_val**2 + y_val**2)],
        ]
    )


def quaternion_to_euler321(quaternion: np.ndarray) -> np.ndarray:
    w_val, x_val, y_val, z_val = normalize_quaternion(quaternion)
    sinr_cosp = 2.0 * (w_val * x_val + y_val * z_val)
    cosr_cosp = 1.0 - 2.0 * (x_val * x_val + y_val * y_val)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w_val * y_val - z_val * x_val)
    pitch = np.sign(sinp) * np.pi / 2.0 if abs(sinp) >= 1.0 else np.arcsin(sinp)

    siny_cosp = 2.0 * (w_val * z_val + x_val * y_val)
    cosy_cosp = 1.0 - 2.0 * (y_val * y_val + z_val * z_val)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


def body_rates_to_quaternion_derivative(quaternion: np.ndarray, body_rates_radps: np.ndarray) -> np.ndarray:
    omega_quat = np.array([0.0, *body_rates_radps])
    return 0.5 * quaternion_multiply(quaternion, omega_quat)


def user_euler_to_internal(euler_rad: np.ndarray) -> np.ndarray:
    return -np.asarray(euler_rad, dtype=float)


def internal_euler_to_user(euler_rad: np.ndarray) -> np.ndarray:
    return -np.asarray(euler_rad, dtype=float)


def user_body_rates_to_internal(body_rates_radps: np.ndarray) -> np.ndarray:
    return -np.asarray(body_rates_radps, dtype=float)


def internal_body_rates_to_user(body_rates_radps: np.ndarray) -> np.ndarray:
    return -np.asarray(body_rates_radps, dtype=float)
