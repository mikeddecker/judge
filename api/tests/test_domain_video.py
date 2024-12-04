# -*- coding: utf-8 -*-
import unittest
from parameterized import parameterized
from domain.folder import Folder
from domain.videoinfo import VideoInfo
from domain.skill import Skill

def generate_empty_strings():
    return [ 
        None, "", " ", "\n", "\r", "  ", "\t ", "\t", " \r\n "
    ]

FOLDER_INSTANCE_VALID = Folder(1, 'competition', None)

class DomainVideoTestSuite(unittest.TestCase):
    """Domain folder test cases."""
    @parameterized.expand([
        (1, "sr-nationals"),
        (2, "sr-nationals"),
        (987, "sr-nationals"),
        (1, "dd3-potatoes"),
        (1, "some-freestyles"),
    ])
    def test_ctor_valid(self, id, name):
        video = VideoInfo(id, name, FOLDER_INSTANCE_VALID)
        self.assertEqual(id, video.Id, f"Video id incorrectly initialized {id}, {video.Id}")
        self.assertEqual(name, video.Name, f"Videoname incorrectly initialized {name}, {video.Name}")
        self.assertEqual(FOLDER_INSTANCE_VALID, video.Folder, f"Videofolder incorrectly initialized")
    
    @parameterized.expand(generate_empty_strings())
    def test_ctor_invalid_name(self, name):
        with self.assertRaises(ValueError):
            VideoInfo(1, name, FOLDER_INSTANCE_VALID)

    @parameterized.expand([None, 0, -1, -55])
    def test_ctor_invalid_id(self, id):
        with self.assertRaises(ValueError):
            VideoInfo(id, "dd3-nationals", None)

    @parameterized.expand([7, "text", True, ValueError])
    def test_ctor_invalid_folder_not_a_folder(self, parent):
        with self.assertRaises(ValueError):
            VideoInfo(1, "dd3-nationals", parent)
    
    # Enforced by Folder
    # def test_ctor_invalid_folder_has_no_Id(self):
    #     with self.assertRaises(AttributeError):
    #         VideoInfo(1, "dd3-nationals", Folder(None, "competition", None))

    def test_change_id_immutable(self):
        with self.assertRaises(AttributeError):
            video = VideoInfo(1, "dd3-nationals", None)
            video.Id = 7

    def test_change_name_immutable(self):
        with self.assertRaises(AttributeError):
            video = VideoInfo(1, "dd3-nationals", None)
            video.Name = "dd3-provincial-pre-round"

    def test_change_folder_immutable(self):
        with self.assertRaises(AttributeError):
            video = VideoInfo(1, "dd3-nationals", FOLDER_INSTANCE_VALID)
            video.Folder = Folder(2, "main", FOLDER_INSTANCE_VALID)

    def test_change_id_private_method(self):
        with self.assertRaises(AttributeError):
            video = VideoInfo(1, "dd3-nationals", None)
            video.__setId(88)

    # Section : No labels
    def test_frames_empty_when_no_labled_frames(self):
        video = VideoInfo(1, "dd3-nationals", None)
        self.assertTrue(len(video.Frames) == 0)


    def test_skills_empty_when_no_labled_skills(self):
        video = VideoInfo(1, "dd3-nationals", None)
        self.assertTrue(len(video.Skills) == 0)
    

    ####################
    # Section : Frames #
    ####################
    def test_add_frame_valid(self):
        pass

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
    def test_add_skill_valid(self):
        video = VideoInfo(1, "dd3-nationals", None)
        skill = Skill()
        video.add_skill(skill)
        self.assertIn(skill, video.Skills, f"Skill is not in property Skills")

    # TODO : add tests
    



if __name__ == '__main__':
    unittest.main()