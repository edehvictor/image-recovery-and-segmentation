"""Portable archive reporting checks; no TensorFlow dependency."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import stage7_summary as summary


class ReportingPaths(unittest.TestCase):
    def test_old_computer_path_resolves_to_local_matching_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root/'run_example'
            run.mkdir()
            record = dict(optimizer='Adam',loss='BCE',learning_rate=0.001)
            (run/'run.json').write_text(json.dumps(record))
            with patch.object(summary,'ROOT',root):
                for saved in ['/Users/other/results/run_example',r'C:\other\results\run_example','run_example']:
                    self.assertEqual(summary.resolve_run_dir(dict(record,run_dir=saved)),run)
                with self.assertRaisesRegex(ValueError,'different loss'):
                    summary.resolve_run_dir(dict(record,run_dir='run_example',loss='Dice'))

    def test_parent_directory_is_rejected(self):
        with self.assertRaises(ValueError):
            summary.resolve_run_dir(dict(run_dir='..'))


if __name__ == '__main__':
    unittest.main()
