import unittest

import cv2
import numpy as np

from camera_capture import add_capture_guide


class CaptureGuideTests(unittest.TestCase):
    def test_guide_is_drawn_on_a_copy(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        preview = add_capture_guide(frame)

        self.assertTrue(np.array_equal(frame, np.zeros_like(frame)))
        self.assertFalse(np.array_equal(preview, frame))
        self.assertEqual(preview.shape, frame.shape)
        self.assertGreater(cv2.countNonZero(cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)), 0)


if __name__ == "__main__":
    unittest.main()
