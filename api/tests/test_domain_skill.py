# -*- coding: utf-8 -*-
import unittest
from parameterized import parameterized
from domain.folder import Folder
from domain.videoinfo import VideoInfo
from domain.skill import Skill
from tests.TestHelper import TestHelper
from domain.enums import DDtype

FOLDER_INSTANCE_VALID = Folder(1, 'competition', None)

# TODO : check tests of whole class

class DomainSkillTestSuite(unittest.TestCase):
    """Domain folder test cases."""
    @parameterized.expand([
        (1, "QU.s.EB.AS.o"),
        (2, "QU.s.EB.AS.o"),
        (987, "QU.s.EB.AS.o"),
        (1, "Gym.cartwheel"),
        (1, "R.1h"),
    ])
    def test_ctor_valid(self, id, dottedName):
        skill = Skill(
            id=1, 
            dottedName='s.EB.AS.o', # optional,
            ddtype = DDtype.DOUBLEDUTCH,
            start=123, end=456)
        assert skill.Id == 1
    
    @parameterized.expand(TestHelper.generate_empty_strings())
    def test_ctor_invalid_name(self, dottedName):
        with self.assertRaises(ValueError):
            Skill(1, dottedName)

    @parameterized.expand(TestHelper.generate_invalid_ids())
    def test_ctor_invalid_id(self, id):
        with self.assertRaises(ValueError):
            Skill(id, "toad", None)

    def test_change_id_immutable(self):
        with self.assertRaises(AttributeError):
            video = Skill(1, "toad")
            video.Id = 7

    def test_change_name_immutable(self):
        with self.assertRaises(AttributeError):
            video = Skill(1, "toad", None)
            video.DottedName = "elephant"

    def test_change_id_private_method(self):
        with self.assertRaises(AttributeError):
            video = VideoInfo(id=1, name="toad", folder=FOLDER_INSTANCE_VALID, frameLength=500, fps=25.5)
            video.__setId(88)



if __name__ == '__main__':
    unittest.main()