# -*- coding: utf-8 -*-
import unittest
from uuid import UUID, uuid4
from parameterized import parameterized
from domain.folder import Folder
from domain.videoinfo import VideoInfo
from domain.skill import Skill
from tests.TestHelper import TestHelper
from domain.enums import DDtype

_FOLDER_UUID = UUID('00000000-0000-0000-0000-000000000001')
FOLDER_INSTANCE_VALID = Folder(_FOLDER_UUID, 'competition', None)

_SKILL_UUID = UUID('00000000-0000-0000-0000-000000000010')

# TODO : check tests of whole class

class DomainSkillTestSuite(unittest.TestCase):
    """Domain folder test cases."""
    @parameterized.expand([
        (UUID('00000000-0000-0000-0000-000000000011'), {}),
        (UUID('00000000-0000-0000-0000-000000000012'), {}),
        (UUID('00000000-0000-0000-0000-000000000013'), {}),
        (UUID('00000000-0000-0000-0000-000000000014'), {}),
        (UUID('00000000-0000-0000-0000-000000000015'), {}),
    ])
    def test_ctor_valid(self, id, info):
        skill = Skill(id=id, skillinfo=info, start=123, end=456)
        assert skill.Id == id
    
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
            skill = Skill(_SKILL_UUID, {}, start=10, end=20)
            skill.Id = uuid4()

    def test_change_info_immutable(self):
        with self.assertRaises(AttributeError):
            skill = Skill(_SKILL_UUID, {}, start=10, end=20)
            skill.SkillInfo = {"Jumper" : {}}

    def test_change_id_private_method(self):
        with self.assertRaises(AttributeError):
            skill = Skill(_SKILL_UUID, {}, start=10, end=20)
            skill.__setId(uuid4())

if __name__ == '__main__':
    print("🧿 Running Domain Skill")
    unittest.main()

