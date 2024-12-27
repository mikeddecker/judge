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
    return VideoInfo(id=1, name="dd3-nationals.mp4", folder=FOLDER_INSTANCE_VALID, frameLength=500)

def get_frame() -> FrameInfo:
    return FrameInfo(frameNr=0, x=0.5, y=0.5, width=0.5, height=0.5)

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
        video = VideoInfo(id=id, name=name, folder=FOLDER_INSTANCE_VALID, frameLength=500)
        self.assertEqual(id, video.Id, f"Video id incorrectly initialized {id}, {video.Id}")
        self.assertEqual(name, video.Name, f"Videoname incorrectly initialized {name}, {video.Name}")
        self.assertEqual(FOLDER_INSTANCE_VALID, video.Folder, f"Videofolder incorrectly initialized")
    
    @parameterized.expand(TestHelper.generate_empty_strings())
    def test_ctor_invalid_name_empty(self, name):
        with self.assertRaises(ValueError):
            VideoInfo(id=1, name=name, folder=FOLDER_INSTANCE_VALID, frameLength=500)

    @parameterized.expand(TestHelper.generate_invalid_strings_only_word_digit_underscore_extensions())
    def test_ctor_invalid_name_word_digits_underscore(self, name):
        with self.assertRaises(ValueError):
            VideoInfo(id=1, name=name, folder=FOLDER_INSTANCE_VALID, frameLength=500)

    @parameterized.expand(TestHelper.generate_invalid_ids())
    def test_ctor_invalid_id(self, id):
        with self.assertRaises(ValueError):
            VideoInfo(id=id, name="dd3-nationals.mp4", folder=FOLDER_INSTANCE_VALID, frameLength=500)

    @parameterized.expand([7, "text", True, ValueError])
    def test_ctor_invalid_folder_not_a_folder(self, parent):
        with self.assertRaises(ValueError):
            VideoInfo(id=1, name="dd3-nationals.mp4", folder=parent, frameLength=500)
    
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
        video.add_framelabel(
            frameNr=fi.FrameNr,
            x=fi.X,
            y=fi.Y,
            width=fi.Width,
            height=fi.Height,
            jumperVisible=fi.JumperVisible,
        )
        self.assertTrue(video.has_frame_been_labeled(fi.FrameNr))
        self.assertEqual(video.Frames[fi.FrameNr], fi)
        self.assertTrue(len(video.Frames) == 1)

    def test_add_frame_valid_multiple(self):
        video = get_video()
        frames = [
            FrameInfo(frameNr=0, x=0.5, y=0.5, width=1.0, height=1.0, jumperVisible=False),
            FrameInfo(frameNr=1, x=0.1, y=0.5, width=0.22, height=0.023),
            FrameInfo(frameNr=2, x=0.3, y=0.5, width=0.22, height=0.23),
            FrameInfo(frameNr=55, x=0.66, y=0.5, width=0.23, height=0.54),
            FrameInfo(frameNr=122, x=0.99, y=0.5, width=0.23, height=0.5),
            FrameInfo(frameNr=450, x=0.3, y=0.44, width=0.5, height=0.5),
            FrameInfo(frameNr=499, x=0.5, y=0.5, width=1.0, height=1.0, jumperVisible=False),
        ]
        for fi in frames:
            video.add_framelabel(
                frameNr=fi.FrameNr,
                x=fi.X,
                y=fi.Y,
                width=fi.Width,
                height=fi.Height,
                jumperVisible=fi.JumperVisible,
            )
        self.assertIn(122, video.Frames.keys())
        self.assertIn(499, video.Frames.keys())
        self.assertEqual(len(video.Frames), len(frames))

    def test_add_frame_invalid_jumperVisible_centered(self):
        video = get_video()
        with self.assertRaises(ValueError):
            video.add_framelabel(
                frameNr=20, jumperVisible=False,
                x=0.55, y=0.55, width=0.8, height=0.4,
            )

    @parameterized.expand([get_video().FrameLength, get_video().FrameLength + 1, get_video().FrameLength * 2])
    def test_add_frame_invalid_frameidx_out_of_bounds(self, invalid_f_nr):
        video = get_video()
        fi = get_frame()
        with self.assertRaises(ValueError):
            video.add_framelabel(
                frameNr=invalid_f_nr,
                x=fi.X,
                y=fi.Y,
                width=fi.Width,
                height=fi.Height,
                jumperVisible=fi.JumperVisible,
            )
    
    ##################################
    # Test frames: has_been_labeled
    ##################################
    def test_has_frame_contained(self):
        pass

    def test_has_frame_not_contained(self):
        pass

    ##################################
    # Test frames: update_framelabel
    ##################################
    def test_update_frame(self):
        # TODO : current framelabels = localization only? 
        # TODO : make decision if update frame only is for videoService & frame, or a 'copy' in video?
        # Update frame = new frame
        pass

    ##################################
    # Test frames: remove framelabel
    ##################################
    def test_remove_frame_valid(self):
        pass

    def test_remove_frame_invalid_not_labeled(self):
        pass

    @parameterized.expand([-123, -1, TestHelper.MAX_FRAMENR + 1])
    def test_remove_frame_invalid_invalid_frameNr(self, invalid_frameId):
        pass
    

    ####################
    # Section : Skills #
    ####################
    def test_skills_empty_when_no_labled_skills(self):
        video = get_video()
        self.assertTrue(len(video.Skills) == 0)
    
    def test_add_skill_valid(self):
        video = get_video()
        skill = Skill(5, "crouger")
        video.add_skill(skill)
        self.assertIn(skill, video.Skills, f"Skill is not in property Skills")

    # TODO : add tests
    

if __name__ == '__main__':
    unittest.main()