# -*- coding: utf-8 -*-
import unittest
from parameterized import parameterized
from domain.folder import Folder
from domain.frameinfo import FrameInfo
from domain.videoinfo import VideoInfo
from tests.TestHelper import TestHelper

FOLDER_INSTANCE_VALID = Folder(id=1, name='competition', parent=None)
MAX_FRAMENR = 65535
VIDEO_CREATOR = lambda: VideoInfo(
    id=1,
    name="dd3-potatoes.mp4",
    folder=FOLDER_INSTANCE_VALID,
    frameLength=500
)
def get_frameinfo(jumperVisible=True) -> FrameInfo:
    return FrameInfo(frameNr=1, x=0.5, y=0.5, width=1.0, height=1.0, jumperVisible=jumperVisible)

class DomainFrameInfoTestSuite(unittest.TestCase):
    """Domain framinfo test cases."""

    ##################################
    # Test constructor
    ##################################
    @parameterized.expand([
        (0, 0.5, 0.5, 0.5, 0.5, True),
        (1, 0.5, 0.5, 0.5, 0.5, True),
        (2, 0.5, 0.5, 0.5, 0.5, True),
        (MAX_FRAMENR, 0.5, 0.5, 0.5, 0.5, True),
        (1, 0.0, 0.5, 0.5, 0.5, True),
        (1, 0.1, 0.5, 0.5, 0.5, True),
        (1, 0.99, 0.5, 0.5, 0.5, True),
        (1, 1.0, 0.5, 0.5, 0.5, True),
        (1, 0.5, 0.0, 0.5, 0.5, True),
        (1, 0.5, 0.1, 0.5, 0.5, True),
        (1, 0.5, 0.99, 0.5, 0.5, True),
        (1, 0.5, 1.0, 0.5, 0.5, True),
        (1, 0.5, 0.5, 0.0, 0.5, True),
        (1, 0.5, 0.5, 0.1, 0.5, True),
        (1, 0.5, 0.5, 0.99, 0.5, True),
        (1, 0.5, 0.5, 1.0, 0.5, True),
        (1, 0.5, 0.5, 0.5, 0.0, True),
        (1, 0.5, 0.5, 0.5, 0.1, True),
        (1, 0.5, 0.5, 0.5, 0.99, True),
        (1, 0.5, 0.5, 0.5, 1.0, True),
        (1, 0.5, 0.5, 1.0, 1.0, False),
    ])
    def test_ctor_valid(self, frameNr: int, x: int, y: int, width: int, height: int, jumperVisible: bool = True):
        fi =FrameInfo(frameNr=frameNr, x=x, y=y, width=width, height=height, jumperVisible=jumperVisible)
        self.assertEqual(frameNr, fi.FrameNr), f"FrameNr not equal"
        self.assertEqual(x, fi.X), f"X not equal"
        self.assertEqual(y, fi.Y), f"Y not equal"
        self.assertEqual(width, fi.Width), f"Width not equal"
        self.assertEqual(height, fi.Height), f"Height not equal"
        self.assertEqual(jumperVisible, fi.JumperVisible), f"JumperVisible not equal"
    
    @parameterized.expand([-123_456, -15, -1, "string", VIDEO_CREATOR])
    def test_ctor_invalid_frameNr_negative_or_no_int(self, invalid_frameNr):
        with self.assertRaises(ValueError):
            FrameInfo(frameNr=invalid_frameNr, x=0.5, y=0.5, height=0.5, width=0.5, jumperVisible=True)

    def test_ctor_invalid_frameNr_to_big(self):
        with self.assertRaises(ValueError):
            FrameInfo(frameNr=MAX_FRAMENR+1, x=0.5, y=0.5, height=0.5, width=0.5, jumperVisible=True)

    @parameterized.expand([
        (-0.01, 0.5, 0.5, 0.5),
        (-0.99, 0.5, 0.5, 0.5),
        (-1.5, 0.5, 0.5, 0.5),
        (+1.1, 0.5, 0.5, 0.5),
        (0.5, -0.01, 0.5, 0.5),
        (0.5, -0.99, 0.5, 0.5),
        (0.5, -1.5, 0.5, 0.5),
        (0.5, +1.1, 0.5, 0.5),
        (0.5, 0.5, -0.01, 0.5),
        (0.5, 0.5, -0.99, 0.5),
        (0.5, 0.5, -1.5, 0.5),
        (0.5, 0.5, +1.1, 0.5),
        (0.5, 0.5, 0.5, -0.01),
        (0.5, 0.5, 0.5, -0.99),
        (0.5, 0.5, 0.5, -1.5),
        (0.5, 0.5, 0.5, +1.1),
    ])
    def test_ctor_invalid_box_values(self, x, y, w, h):
        with self.assertRaises(ValueError):
            FrameInfo(frameNr=0, x=x, y=y, width=w, height=h)
    
    @parameterized.expand([
        (0.2, 0.5, 0.5, 0.5),
        (0.5, 0.2, 0.5, 0.5),
        (0.5, 0.5, 0.9, 0.5),
        (0.5, 0.5, 0.5, 0.2),
    ])
    def test_ctor_invalid_jumperVisible_false_so_full_view(self, x, y, w, h):
        with self.assertRaises(ValueError):
            FrameInfo(
                frameNr=20, jumperVisible=False,
                x=x, y=y, width=w, height=h,
            )


    #############################################
    # Test immutable properties & private method
    #############################################
    def test_change_frameNr_immutable(self):
        fi = FrameInfo(frameNr=1, x=0.5, y=0.5, height=0.5, width=0.5, jumperVisible=True)
        with self.assertRaises(AttributeError):
            fi.__setFrameNr(5)


    ##################################
    # Test setters
    ##################################
    @parameterized.expand(TestHelper.generate_zero_to_one_included())
    def test_setX_valid(self, x):
        fi = get_frameinfo()
        fi.setX(x)
        self.assertEqual(x, fi.X)

    @parameterized.expand(TestHelper.generate_zero_to_one_included())
    def test_setY_valid(self, y):
        fi = get_frameinfo()
        fi.setY(y)
        self.assertEqual(y, fi.Y)

    @parameterized.expand(TestHelper.generate_zero_to_one_included())
    def test_setWidth_valid(self, width):
        fi = get_frameinfo()
        fi.setWidth(width)
        self.assertEqual(width, fi.Width)

    @parameterized.expand(TestHelper.generate_zero_to_one_included())
    def test_setHeight_valid(self, height):
        fi = get_frameinfo()
        fi.setHeight(height)
        self.assertEqual(height, fi.Height)

    @parameterized.expand(TestHelper.generate_zero_to_one_included(valid=False))
    def test_setX_invalid(self, x):
        fi = get_frameinfo()
        with self.assertRaises(ValueError):
            fi.setX(x)

    @parameterized.expand(TestHelper.generate_zero_to_one_included(valid=False))
    def test_setY_invalid(self, y):
        fi = get_frameinfo()
        with self.assertRaises(ValueError):
            fi.setY(y)

    @parameterized.expand(TestHelper.generate_zero_to_one_included(valid=False))
    def test_setWidth_invalid(self, width):
        fi = get_frameinfo()
        with self.assertRaises(ValueError):
            fi.setWidth(width)

    @parameterized.expand(TestHelper.generate_zero_to_one_included(valid=False))
    def test_setHeight_invalid(self, height):
        fi = get_frameinfo()
        with self.assertRaises(ValueError):
            fi.setHeight(height)

    def test_setX_invalid_jumperNotVisible_so_centered(self):
        fi = get_frameinfo(jumperVisible=False)
        with self.assertRaises(ValueError):
            fi.setX(0.2)

    def test_setY_invalid_jumperNotVisible_so_centered(self):
        fi = get_frameinfo(jumperVisible=False)
        with self.assertRaises(ValueError):
            fi.setY(0.2)

    def test_setWidth_invalid_jumperNotVisible_so_centered(self):
        fi = get_frameinfo(jumperVisible=False)
        with self.assertRaises(ValueError):
            fi.setWidth(0.2)

    def test_setHeight_invalid_jumperNotVisible_so_centered(self):
        fi = get_frameinfo(jumperVisible=False)
        with self.assertRaises(ValueError):
            fi.setHeight(0.2)

    def test_setJumperVisible_False_check_centered_valid(self):
        fi = get_frameinfo(jumperVisible=True)
        self.assertTrue(0.5 == fi.X)
        self.assertTrue(0.5 == fi.Y)
        self.assertTrue(1.0 == fi.Width)
        self.assertTrue(1.0 == fi.Height)
        fi.setJumperVisible(False)
        self.assertFalse(fi.JumperVisible)

    @parameterized.expand([
        (0.2, 0.5, 0.5, 0.5),
        (0.5, 0.2, 0.5, 0.5),
        (0.5, 0.5, 0.9, 0.5),
        (0.5, 0.5, 0.5, 0.2),
    ])
    def test_setJumperVisible_False_check_centered_invalid(self, x, y, w, h):
        fi = FrameInfo(frameNr=0, x=x, y=y, width=w, height=h, jumperVisible=True)
        with self.assertRaises(ValueError):
            fi.setJumperVisible(False)
    

if __name__ == '__main__':
    unittest.main()