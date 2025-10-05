import numpy as np
from collections import deque
import cv2


class VideoEnhancer:
    def __init__(
        self,
        enable=True,
        clahe=True,
        sharpen=True,
        gamma=1.5,
        wb=True,
        temporal_window=5,  # odd number (0/1 disables)
        stabilize=False,  # ECC single pass
        ecc_iters=15,
        ecc_eps=1e-4,
    ):
        self.enable = enable
        self.clahe = clahe
        self.sharpen = sharpen
        self.gamma = gamma
        self.wb = wb
        self.temporal_window = temporal_window if temporal_window >= 3 else 0
        self.buffer = deque(maxlen=self.temporal_window or 1)
        self.stabilize = stabilize
        self.prev_gray = None
        self.M = np.eye(3, dtype=np.float32)
        self.ecc_iters = ecc_iters
        self.ecc_eps = ecc_eps
        self._clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(5, 5))

    def _white_balance(self, bgr):
        # Simple gray-world WB
        avgB, avgG, avgR = [bgr[..., i].mean() for i in range(3)]
        kb, kg, kr = (
            (avgB + avgG + avgR) / (3 * avgB + 1e-6),
            (avgB + avgG + avgR) / (3 * avgG + 1e-6),
            (avgB + avgG + avgR) / (3 * avgR + 1e-6),
        )
        wb = bgr.copy().astype(np.float32)
        wb[..., 0] *= kb
        wb[..., 1] *= kg
        wb[..., 2] *= kr
        return np.clip(wb, 0, 255).astype(np.uint8)

    def _apply_clahe(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = self._clahe.apply(v)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _apply_gamma(self, bgr):
        if abs(self.gamma - 1.0) < 1e-3:
            return bgr
        inv = 1.0 / self.gamma
        lut = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(bgr, lut)

    def _sharpen(self, bgr):
        # Unsharp mask
        blur = cv2.GaussianBlur(bgr, (0, 0), 1.0)
        return cv2.addWeighted(bgr, 1.5, blur, -0.5, 0)

    def _stabilize(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame
        # ECC affine (fast & robust for small shakes)
        warp_mode = cv2.MOTION_AFFINE
        warp_mat = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp_mat = cv2.findTransformECC(
                self.prev_gray,
                gray,
                warp_mat,
                warp_mode,
                (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    self.ecc_iters,
                    self.ecc_eps,
                ),
            )
            stabilized = cv2.warpAffine(
                frame,
                warp_mat,
                (frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
            self.prev_gray = cv2.warpAffine(
                gray,
                warp_mat,
                (gray.shape[1], gray.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            )
            return stabilized
        except cv2.error:
            self.prev_gray = gray
            return frame

    def enhance(self, frame_bgr):
        if not self.enable:
            return frame_bgr

        img = frame_bgr

        # Optional stabilization first
        if self.stabilize:
            img = self._stabilize(img)

        # WB → CLAHE → Gamma → Sharpen
        if self.wb:
            img = self._white_balance(img)
        if self.clahe:
            img = self._apply_clahe(img)
        if self.gamma and abs(self.gamma - 1.0) > 1e-3:
            img = self._apply_gamma(img)
        if self.sharpen:
            img = self._sharpen(img)

        # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # l, a, b = cv2.split(lab)
        # l_denoised = bm3d(l / 255.0, sigma_psd=0.08, profile="vn")  # tune sigma
        # l_denoised = (np.clip(l_denoised, 0, 1) * 255).astype(np.uint8)
        # img = cv2.cvtColor(cv2.merge([l_denoised, a, b]), cv2.COLOR_LAB2BGR)

        # Temporal denoise (running median on small window)
        if self.temporal_window:
            self.buffer.append(img.astype(np.uint8))
            # On first few frames, just average available
            stack = np.stack(list(self.buffer), axis=0)
            img = np.median(stack, axis=0).astype(np.uint8)

        return img
