# -*- coding: utf-8 -*-
import unittest
from parameterized import parameterized
from domain.folder import Folder
from domain.frameinfo import FrameInfo
from domain.videoinfo import VideoInfo
from domain.skill import Skill
from tests.TestHelper import TestHelper

FOLDER_INSTANCE_VALID = Folder(id=1, name='competition', parent=None)
def get_video() -> VideoInfo:
    return VideoInfo(id=1, name="dd3-nationals.mp4", folder=FOLDER_INSTANCE_VALID, frameLength=500, fps=25.2, duration=12.3, width=1920, height=1080)

def get_frame() -> FrameInfo:
    return FrameInfo(frameNr=0, x=0.5, y=0.5, width=0.5, height=0.5)

def test_get_relative_video_path(self):
    video = get_video()
    expected_path = os.path.join("competition", "dd3-nationals.mp4")
    self.assertEqual(video.get_relative_video_path(), expected_path)

def test_get_image_path(self):
    video = get_video()
    # TODO: Update when image_path is implemented
    self.assertEqual(video.get_image_path(), "/home/miked/Pictures/Screenshots/dd3.png")

def test_fps_property(self):
    video = get_video()
    self.assertEqual(video.FPS, 25.2)

def test_get_duration(self):
    video = get_video()
    expected_duration = 500 / 25.2
    self.assertAlmostEqual(video.get_duration(), expected_duration)

class DomainVideoTestSuite(unittest.TestCase):
    """Domain folder test cases."""

    ##################################
    # Test constructor
    ##################################
    @parameterized.expand([
        (1, "sr-nationals.mp4"),
        (2, "sr-nationals.mp4"),
        (987, "sr-nationals.mp4"),
        (1, "dd3-potatoes.mp4"),
        (1, "some-freestyles.mp4"),
    ])
    def test_ctor_valid(self, id, name):
        video = VideoInfo(id=id, name=name, folder=FOLDER_INSTANCE_VALID, frameLength=500, fps=25.2, duration=12.3, width=1920, height=1080)
        self.assertEqual(id, video.Id, f"Video id incorrectly initialized {id}, {video.Id}")
        self.assertEqual(name, video.Name, f"Videoname incorrectly initialized {name}, {video.Name}")
        self.assertEqual(FOLDER_INSTANCE_VALID, video.Folder, f"Videofolder incorrectly initialized")
    
    @parameterized.expand(TestHelper.generate_empty_strings())
    def test_ctor_invalid_name_empty(self, name):
        with self.assertRaises(ValueError):
            VideoInfo(id=1, name=name, folder=FOLDER_INSTANCE_VALID, frameLength=500, fps=25.2, duration=12.3, width=1920, height=1080)

    @parameterized.expand(TestHelper.generate_invalid_strings_only_word_digit_underscore_extensions())
    def test_ctor_invalid_name_word_digits_underscore(self, name):
        with self.assertRaises(ValueError):
            VideoInfo(id=1, name=name, folder=FOLDER_INSTANCE_VALID, frameLength=500, fps=25.2, duration=12.3, width=1920, height=1080)

    @parameterized.expand(TestHelper.generate_invalid_ids())
    def test_ctor_invalid_id(self, id):
        with self.assertRaises(ValueError):
            VideoInfo(id=id, name="dd3-nationals.mp4", folder=FOLDER_INSTANCE_VALID, frameLength=500, fps=25.2, duration=12.3, width=1920, height=1080)

    @parameterized.expand([7, "text", True, ValueError])
    def test_ctor_invalid_folder_not_a_folder(self, parent):
        with self.assertRaises(ValueError):
            VideoInfo(id=1, name="dd3-nationals.mp4", folder=parent, frameLength=500, fps=25.2, duration=12.3, width=1920, height=1080)
    
    #############################################
    # Test immutable properties & private method
    #############################################
    def test_change_id_immutable(self):
        with self.assertRaises(AttributeError):
            video = get_video()
            video.Id = 7

    def test_change_name_immutable(self):
        with self.assertRaises(AttributeError):
            video = get_video()
            video.Name = "dd3-provincial-pre-round"

    def test_change_folder_immutable(self):
        with self.assertRaises(AttributeError):
            video = get_video()
            video.Folder = Folder(id=2, name="main", parent=FOLDER_INSTANCE_VALID)

    def test_change_id_private_method(self):
        with self.assertRaises(AttributeError):
            video = get_video()
            video.__setId(88)

    ##################################
    # Test frames: ADD
    ##################################
    def test_frames_empty_when_no_labled_frames(self):
        video = get_video()
        self.assertTrue(len(video.Frames) == 0)

    def test_add_frame_valid(self):
        video = get_video()
        fi = get_frame()
        video.add_framelabel(label=fi)
        self.assertTrue(video.has_frame_been_labeled(fi))
        self.assertEqual(video.Frames[fi.FrameNr], fi)
        self.assertTrue(len(video.Frames) == 1)

    def test_add_frame_valid_multiple(self):
        video = get_video()
        frames = [
            FrameInfo(frameNr=0, x=0.5, y=0.5, width=1.0, height=1.0, jumperVisible=False),
            FrameInfo(frameNr=1, x=0.1, y=0.5, width=0.22, height=0.073),
            FrameInfo(frameNr=2, x=0.3, y=0.5, width=0.22, height=0.23),
            FrameInfo(frameNr=55, x=0.66, y=0.5, width=0.23, height=0.54),
            FrameInfo(frameNr=122, x=0.99, y=0.5, width=0.23, height=0.5),
            FrameInfo(frameNr=450, x=0.3, y=0.44, width=0.5, height=0.5),
            FrameInfo(frameNr=499, x=0.5, y=0.5, width=1.0, height=1.0, jumperVisible=False),
        ]
        for fi in frames:
            video.add_framelabel(label=fi)

        self.assertTrue(video.contains_frame_with_number(122)),
        self.assertTrue(video.contains_frame_with_number(499))
        self.assertEqual(len(video.Frames), len(frames))

    def test_add_frame_invalid_jumperVisible_centered(self):
        video = get_video()
        with self.assertRaises(ValueError):
            video.add_framelabel(FrameInfo(
                frameNr=20, jumperVisible=False,
                x=0.55, y=0.55, width=0.8, height=0.4,
            ))

    @parameterized.expand([get_video().FrameLength, get_video().FrameLength + 1, get_video().FrameLength * 2])
    def test_add_frame_invalid_frameidx_out_of_bounds(self, invalid_f_nr):
        video = get_video()
        fi = get_frame()
        fi = FrameInfo(invalid_f_nr, x=fi.X, y=fi.Y, width=fi.Width, height=fi.Height, jumperVisible=fi.JumperVisible)
        with self.assertRaises(ValueError):
            video.add_framelabel(fi)
    
    @parameterized.expand([True, get_video(), lambda x: [x]])
    def test_add_frame_invalid_not_a_frameinfo(self, not_a_label):
        video = get_video()
        with self.assertRaises(ValueError):
            video.add_framelabel(label=not_a_label)

    ##################################
    # Test frames: has_been_labeled
    ##################################
    @parameterized.expand([0,1,20,235])
    def test_has_frame_contained(self, nr):
        video = get_video()
        fi = get_frame()
        fi = FrameInfo(frameNr=nr, x=fi.X, y=fi.Y, width=fi.Width, height=fi.Height, jumperVisible=fi.JumperVisible)
        video.add_framelabel(fi)
        self.assertTrue(video.has_frame_been_labeled(fi))
        self.assertTrue(video.contains_frame_with_number(nr))

    @parameterized.expand([0,1,20,1235])
    def test_has_frame_not_contained(self, nr):
        video = get_video()
        fi = get_frame()
        fi = FrameInfo(frameNr=nr, x=fi.X, y=fi.Y, width=fi.Width, height=fi.Height, jumperVisible=fi.JumperVisible)
        self.assertFalse(video.has_frame_been_labeled(fi))
        self.assertFalse(video.contains_frame_with_number(nr))

    ##################################
    # Test frames: remove framelabel
    ##################################
    def test_remove_frame_valid(self):
        vid = get_video()
        fi = get_frame()
        vid.add_framelabel(fi)
        self.assertTrue(len(vid.Frames) == 1)
        vid.remove_framelabel(fi.FrameNr, label=fi)
        self.assertTrue(len(vid.Frames) == 0)

    def test_remove_frame_invalid_not_labeled(self):
        vid : VideoInfo = get_video()
        fi : FrameInfo = get_frame()
        vid.add_framelabel(fi)

        with self.assertRaises(ValueError):
            vid.remove_framelabel(fi.FrameNr + 20, label=FrameInfo(
                frameNr=fi.FrameNr,
                x = 0.4, y = 0.6, width= 0.6, height=0.33, 
                labeltype = 1
            ))

    @parameterized.expand([-123, -1, TestHelper.MAX_FRAMENR + 1])
    def test_remove_frame_invalid_invalid_frameNr(self, invalid_frameId):
        vid = get_video()
        with self.assertRaises(ValueError):
            vid.remove_framelabel(frameNr=invalid_frameId, label={})

    ####################
    # Section : Skills #
    ####################
    def test_skills_empty_when_no_labled_skills(self):
        video = get_video()
        self.assertTrue(len(video.Skills) == 0)
    
    def test_add_skill_valid(self):
        video = get_video()
        skill = Skill(5, {"name": "crouger"}, start=10, end=20)
        video.add_skill(skill)
        self.assertIn(skill, video.Skills, f"Skill is not in property Skills")

    def test_add_skill_duplicate(self):
        video = get_video()
        skill = Skill(5, {"name": "crouger"}, start=10, end=20)
        video.add_skill(skill)
        with self.assertRaises(ValueError):
            video.add_skill(skill)

    def test_add_skill_overlap(self):
        video = get_video()
        skill1 = Skill(5, {"name": "crouger"}, start=10, end=20)
        skill2 = Skill(6, {"name": "another"}, start=15, end=25)
        video.add_skill(skill1)
        with self.assertRaises(ValueError):
            video.add_skill(skill2)

    def test_get_skill_existing(self):
        video = get_video()
        skill = Skill(5, {"name": "crouger"}, start=10, end=20)
        video.add_skill(skill)
        retrieved = video.get_skill(10, 20)
        self.assertEqual(retrieved, skill)

    def test_get_skill_non_existing(self):
        video = get_video()
        retrieved = video.get_skill(10, 20)
        self.assertIsNone(retrieved)

    def test_remove_skill(self):
        video = get_video()
        skill = Skill(5, {"name": "crouger"}, start=10, end=20)
        video.add_skill(skill)
        video.remove_skill(skill)
        self.assertNotIn(skill, video.Skills)
    
if __name__ == '__main__':
    print("🧿 Running Domain Video")
    unittest.main()

