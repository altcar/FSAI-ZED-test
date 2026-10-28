"""Geometry and fair-comparison checks; no GPU or SDK required."""
import ctypes
import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare import evaluate
from prepare_stereo import rotate_depth
from sim_streamer import StreamingParameters


class ComparisonTests(unittest.TestCase):
    def test_common_support_and_missing_depth(self):
        gt = np.array([[1, 2, 3, 100, np.nan]], dtype=np.float32)
        net = np.array([[2, 3, 4, 1, 1]], dtype=np.float32)
        sdk = np.array([[1, np.nan, 5, 1, 1]], dtype=np.float32)
        report, target, masks, common = evaluate(gt, {'net': net, 'sdk': sdk})
        self.assertEqual(report['target_pixels'], 3)
        self.assertEqual(report['common_pixels'], 2)
        self.assertAlmostEqual(report['pairwise']['sdk']['coverage_of_gt'], 2/3)
        self.assertEqual(report['common_pixel_metrics']['net']['mae_m'], 1)
        self.assertEqual(report['common_pixel_metrics']['sdk']['mae_m'], 1)

    def test_mismatched_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(np.ones((3, 4)), {'wrong': np.ones((4, 3))})

    def test_rotated_plane_optical_z(self):
        # A frontoparallel 10m plane in the source camera has nonconstant Z after yaw.
        k = np.array([[80., 0, 40], [0, 80, 30], [0, 0, 1]])
        rotation, _ = cv2.Rodrigues(np.array([0., 0.05, 0.]))
        z = rotate_depth(np.full((60, 80), 10, dtype=np.float32), k, rotation, k, (80, 60))
        for u, v in [(20, 25), (40, 30), (60, 35)]:
            q = rotation.T @ np.linalg.inv(k) @ [u, v, 1]
            self.assertAlmostEqual(float(z[v, u]), 10 / q[2], places=5)
        self.assertGreater(abs(float(z[30, 20]) - float(z[30, 60])), 0.1)

    def test_c_abi_layout(self):
        # Checked independently against sizeof/offsetof of bundled C++ types_c.h.
        self.assertEqual(ctypes.sizeof(StreamingParameters), 100)
        self.assertEqual(StreamingParameters.port.offset, 44)
        self.assertEqual(StreamingParameters.input_format.offset, 60)
        self.assertEqual(StreamingParameters.depth_bitrate.offset, 96)


if __name__ == '__main__':
    unittest.main()
