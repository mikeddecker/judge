# -*- coding: utf-8 -*-
import unittest
from parameterized import parameterized
from domain.folder import Folder

def generate_empty_strings():
    return [ 
        None, "", " ", "\n", "\r", "  ", "\t ", "\t", " \r\n "
    ]
class DomainFolderTestSuite(unittest.TestCase):
    """Domain folder test cases."""
    @parameterized.expand([
        (1, "competition", "main"),
        (2, "competition", "main"),
        (125, "competition", "main"),
        (125, "free", "main"),
        (125, "competition", None),
    ])
    def test_create_folder_valid(self, id, name, parentname):
        folderparent = Folder(id, parentname, None) if parentname else None
        folder = Folder(id, name, folderparent)
        self.assertEqual(name, folder.Name, f"Foldername incorrectly initialized {name}, {folder.Name}")
        self.assertEqual(id, folder.Id, f"Folder id incorrectly initialized {id}, {folder.Id}")
        self.assertEqual(folderparent, folder.Parent, f"Parent of folder incorrectly initialized")

    @parameterized.expand(generate_empty_strings())
    def test_create_folder_invalid_name_none(self, name):
        with self.assertRaises(ValueError):
            Folder(1, name, None)

    @parameterized.expand([None, 0, -1, -55])
    def test_create_folder_invalid_id(self, id):
        with self.assertRaises(ValueError):
            Folder(id, "competition", None)

    @parameterized.expand([7, "text", ValueError])
    def test_create_folder_invalid_parent_not_a_folder(self, parent):
        with self.assertRaises(ValueError):
            Folder(1, "competition", parent)

    def test_change_id_immutable(self):
        with self.assertRaises(AttributeError):
            folder = Folder(1, "competition", None)
            folder.Id = 7

    def test_change_name_immutable(self):
        with self.assertRaises(AttributeError):
            folder = Folder(1, "competition", None)
            folder.Name = "free"

    def test_change_parent_immutable(self):
        with self.assertRaises(AttributeError):
            folder = Folder(1, "competition", None)
            folder.Parent = Folder(2, "main", None)

if __name__ == '__main__':
    unittest.main()