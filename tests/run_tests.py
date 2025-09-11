"""
Test runner for corridor segmentation tests.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_all_tests():
    """Run all unit tests."""
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(str(start_dir), pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)