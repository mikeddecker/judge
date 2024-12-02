# -*- coding: utf-8 -*-
import unittest
from parameterized import parameterized
from domain.folder import Folder
from domain.video import Video

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
        video = Video(id, name, FOLDER_INSTANCE_VALID)
        self.assertEqual(id, video.Id, f"Video id incorrectly initialized {id}, {video.Id}")
        self.assertEqual(name, video.Name, f"Videoname incorrectly initialized {name}, {video.Name}")
        self.assertEqual(FOLDER_INSTANCE_VALID, video.Folder, f"Videofolder incorrectly initialized")
    
    @parameterized.expand(generate_empty_strings())
    def test_ctor_invalid_name(self, name):
        with self.assertRaises(ValueError):
            Video(1, name, FOLDER_INSTANCE_VALID)

    @parameterized.expand([None, 0, -1, -55])
    def test_ctor_invalid_id(self, id):
        with self.assertRaises(ValueError):
            Video(id, "dd3-nationals", None)

    @parameterized.expand([7, "text", True, ValueError])
    def test_ctor_invalid_folder_not_a_folder(self, parent):
        with self.assertRaises(ValueError):
            Video(1, "dd3-nationals", parent)
    
    # Enforced by Folder
    # def test_ctor_invalid_folder_has_no_Id(self):
    #     with self.assertRaises(AttributeError):
    #         Video(1, "dd3-nationals", Folder(None, "competition", None))

    def test_change_id_immutable(self):
        with self.assertRaises(AttributeError):
            video = Video(1, "dd3-nationals", None)
            video.Id = 7

    def test_change_name_immutable(self):
        with self.assertRaises(AttributeError):
            video = Video(1, "dd3-nationals", None)
            video.Name = "dd3-provincial-pre-round"

    def test_change_folder_immutable(self):
        with self.assertRaises(AttributeError):
            video = Video(1, "dd3-nationals", FOLDER_INSTANCE_VALID)
            video.Folder = Folder(2, "main", FOLDER_INSTANCE_VALID)

if __name__ == '__main__':
    unittest.main()