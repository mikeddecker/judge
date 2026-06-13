# -*- coding: utf-8 -*-
"""
Data-driven unit tests for FrameInfo (frame labels).

Test cases are loaded from all *.json files in:
    tests/testdata/framelabels/

Each JSON file must have the structure:
    {
        "description": "...",
        "valid": [
            {
                "frameNr": <int>,
                "x": <float 0..1>,
                "y": <float 0..1>,
                "width": <float 0..1>,
                "height": <float 0..1>,
                "jumperVisible": <bool>,   (optional, default true)
                "labeltype": <int>,        (optional, default 0)
                "description": "..."       (optional, human-readable label)
            },
            ...
        ],
        "invalid": [
            {
                ... same fields as valid ...,
                "expected_error": "ValueError",
                "description": "..."
            },
            ...
        ]
    }

To add more test cases, create a new *.json file in the testdata/framelabels/
directory or extend an existing one. No Python changes are required.
"""
import glob
import json
import os
import unittest

from domain.frameinfo import FrameInfo

_TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "testdata", "framelabels")


def _load_all_cases():
    """Load valid and invalid cases from all *.json files in the testdata directory."""
    valid_cases = []
    invalid_cases = []

    for filepath in sorted(glob.glob(os.path.join(_TESTDATA_DIR, "*.json"))):
        filename = os.path.basename(filepath)
        with open(filepath, encoding="utf-8") as fh:
            data = json.load(fh)

        for idx, case in enumerate(data.get("valid", [])):
            label = f"{filename}[{idx}]: {case.get('description', '')}"
            valid_cases.append((label, case))

        for idx, case in enumerate(data.get("invalid", [])):
            label = f"{filename}[{idx}]: {case.get('description', '')}"
            invalid_cases.append((label, case))

    return valid_cases, invalid_cases


def _build_frameinfo(case: dict) -> FrameInfo:
    return FrameInfo(
        frameNr=case["frameNr"],
        x=case["x"],
        y=case["y"],
        width=case["width"],
        height=case["height"],
        jumperVisible=case.get("jumperVisible", True),
        labeltype=case.get("labeltype", 0),
    )


_VALID_CASES, _INVALID_CASES = _load_all_cases()


class FrameLabelValidJsonDataTest(unittest.TestCase):
    """Runs every 'valid' entry from the JSON test-data files as a sub-test."""

    def test_valid_cases(self):
        """All valid cases from testdata/framelabels/*.json must construct without error."""
        self.assertTrue(
            len(_VALID_CASES) > 0,
            "No valid test cases found in testdata/framelabels/. "
            "Add *.json files with a 'valid' array to cover more label scenarios.",
        )
        for label, case in _VALID_CASES:
            with self.subTest(case=label):
                fi = _build_frameinfo(case)
                self.assertEqual(fi.FrameNr, case["frameNr"])
                self.assertAlmostEqual(fi.X, case["x"])
                self.assertAlmostEqual(fi.Y, case["y"])
                self.assertAlmostEqual(fi.Width, case["width"])
                self.assertAlmostEqual(fi.Height, case["height"])
                self.assertEqual(fi.JumperVisible, case.get("jumperVisible", True))
                self.assertEqual(fi.LabelType, case.get("labeltype", 0))


class FrameLabelInvalidJsonDataTest(unittest.TestCase):
    """Runs every 'invalid' entry from the JSON test-data files as a sub-test."""

    def test_invalid_cases(self):
        """All invalid cases from testdata/framelabels/*.json must raise ValueError."""
        self.assertTrue(
            len(_INVALID_CASES) > 0,
            "No invalid test cases found in testdata/framelabels/. "
            "Add *.json files with an 'invalid' array to cover rejection scenarios.",
        )
        for label, case in _INVALID_CASES:
            with self.subTest(case=label):
                with self.assertRaises(
                    ValueError,
                    msg=f"Expected ValueError for case: {label}",
                ):
                    _build_frameinfo(case)


if __name__ == "__main__":
    unittest.main()
