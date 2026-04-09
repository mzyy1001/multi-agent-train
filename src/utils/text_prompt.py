from __future__ import annotations

import numpy as np


def speaker_obs_to_text(obs: np.ndarray) -> str:
    """Convert speaker (3,) observation (goal landmark color) to text prompt."""
    r, g, b = obs[0], obs[1], obs[2]
    return (
        f"Speaker agent. Target landmark color: R={r:.2f} G={g:.2f} B={b:.2f}. "
        "Send a message to guide the listener."
    )


def listener_obs_to_text(obs: np.ndarray) -> str:
    """Convert listener (11,) observation to text prompt.

    obs layout: [vel_x, vel_y, lm1_dx, lm1_dy, lm2_dx, lm2_dy, lm3_dx, lm3_dy, c1, c2, c3]
    We include velocity and landmark positions but not the env's comm channel (c1-c3),
    since we use our own differentiable channel.
    """
    vel = obs[:2]
    landmarks = obs[2:8].reshape(3, 2)
    return (
        f"Listener agent. Velocity: [{vel[0]:.2f}, {vel[1]:.2f}]. "
        f"Landmarks: A=[{landmarks[0, 0]:.2f}, {landmarks[0, 1]:.2f}], "
        f"B=[{landmarks[1, 0]:.2f}, {landmarks[1, 1]:.2f}], "
        f"C=[{landmarks[2, 0]:.2f}, {landmarks[2, 1]:.2f}]. "
        "Move to the target landmark."
    )


def centralized_obs_to_text(
    speaker_obs: np.ndarray,
    listener_obs: np.ndarray,
) -> str:
    """Combine both observations for centralized baseline."""
    r, g, b = speaker_obs[0], speaker_obs[1], speaker_obs[2]
    vel = listener_obs[:2]
    landmarks = listener_obs[2:8].reshape(3, 2)
    return (
        f"Centralized agent. Target color: R={r:.2f} G={g:.2f} B={b:.2f}. "
        f"Listener velocity: [{vel[0]:.2f}, {vel[1]:.2f}]. "
        f"Landmarks: A=[{landmarks[0, 0]:.2f}, {landmarks[0, 1]:.2f}], "
        f"B=[{landmarks[1, 0]:.2f}, {landmarks[1, 1]:.2f}], "
        f"C=[{landmarks[2, 0]:.2f}, {landmarks[2, 1]:.2f}]. "
        "Move to the target landmark."
    )
