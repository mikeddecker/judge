# -*- coding: utf-8 -*-
import unittest
from parameterized import parameterized
from domain.folder import Folder
from domain.videoinfo import VideoInfo
from domain.skill import Skill
from tests.TestHelper import TestHelper

FOLDER_INSTANCE_VALID = Folder(id=1, name='competition', parent=None)

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
            video = VideoInfo(id=1, name="dd3-nationals.mp4", folder=FOLDER_INSTANCE_VALID, frameLength=500)
            video.Id = 7

    def test_change_name_immutable(self):
        with self.assertRaises(AttributeError):
            video = VideoInfo(id=1, name="dd3-nationals.mp4", folder=FOLDER_INSTANCE_VALID, frameLength=500)
            video.Name = "dd3-provincial-pre-round"

    def test_change_folder_immutable(self):
        with self.assertRaises(AttributeError):
            video = VideoInfo(id=1, name="dd3-nationals.mp4", folder=FOLDER_INSTANCE_VALID, frameLength=500)
            video.Folder = Folder(id=2, name="main", parent=FOLDER_INSTANCE_VALID)

    def test_change_id_private_method(self):
        with self.assertRaises(AttributeError):
            video = VideoInfo(id=1, name="dd3-nationals.mp4", folder=FOLDER_INSTANCE_VALID, frameLength=500)
            video.__setId(88)


    ##################################
    # Test frames
    ##################################
    def test_frames_empty_when_no_labled_frames(self):
        video = VideoInfo(id=1, name="dd3-nationals.mp4", folder=FOLDER_INSTANCE_VALID, frameLength=500)
        self.assertTrue(len(video.Frames) == 0)

    def test_add_frame_valid(self):
        video = VideoInfo(id=1, name="dd3-nationals.mp4", folder=FOLDER_INSTANCE_VALID, frameLength=500)
        self.assertTrue(NotImplementedError)

    def test_add_frame_invalid_none(self):
        pass

    def test_add_frame_invalid_duplicate(self):
        pass

    def test_add_frame_invalid_frameidx_already_exists(self):
        pass

    def test_add_frame_invalid_frameidx_out_of_bounds(self):
        # TODO : e.g. video is 3080 frames, but frame has idx 3080 or 5555
        pass

    def test_has_frame_contained(self):
        pass

    def test_has_frame_not_contained(self):
        pass

    def test_get_frame_valid(self):
        pass

    def test_get_frame_invalid_not_contained_none(self):
        # TODO : when frameidx not contained, return None or KeyError
        pass

    def test_update_frame(self):
        # TODO : current framelabels = localization only? 
        # TODO : make decision if update frame only is for videoService & frame, or a 'copy' in video?
        pass

    def test_remove_frame_valid(self):
        pass

    def test_remove_frame_invalid_not_in_dict(self):
        pass
    

    ####################
    # Section : Skills #
    ####################
    def test_skills_empty_when_no_labled_skills(self):
        video = VideoInfo(id=1, name="dd3-nationals.mp4", folder=FOLDER_INSTANCE_VALID, frameLength=500)
        self.assertTrue(len(video.Skills) == 0)
    
    def test_add_skill_valid(self):
        video = VideoInfo(id=1, name="dd3-nationals.mp4", folder=FOLDER_INSTANCE_VALID, frameLength=500)
        skill = Skill(5, "crouger")
        video.add_skill(skill)
        self.assertIn(skill, video.Skills, f"Skill is not in property Skills")

    # TODO : add tests
    



if __name__ == '__main__':
    unittest.main()