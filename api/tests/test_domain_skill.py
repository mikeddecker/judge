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
        (1, {}),
        (2, {}),
        (987, {}),
        (1, {}),
        (1, {}),
    ])
    def test_ctor_valid(self, id, info):
        skill = Skill(id=1, skillinfo=info, start=123, end=456)
        assert skill.Id == 1
    
    # @parameterized.expand(TestHelper.generate_empty_strings())
    # def test_ctor_invalid_name(self, empty_string):
    #     with self.assertRaises(ValueError):
    #         Skill(1, empty_string)

    @parameterized.expand(TestHelper.generate_invalid_ids())
    def test_ctor_invalid_id(self, id):
        with self.assertRaises(ValueError):
            Skill(id, {}, start=10, end=20)

    def test_change_id_immutable(self):
        with self.assertRaises(AttributeError):
            skill = Skill(1, {}, start=10, end=20)
            skill.Id = 7

    def test_change_info_immutable(self):
        with self.assertRaises(AttributeError):
            skill = Skill(1, {}, start=10, end=20)
            skill.SkillInfo = {"Jumper" : {}}

    def test_change_id_private_method(self):
        with self.assertRaises(AttributeError):
            skill = Skill(1, {}, start=10, end=20)
            skill.__setId(88)

if __name__ == '__main__':
    print("🧿 Running Domain Skill")
    unittest.main()

