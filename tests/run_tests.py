#!/usr/bin/env python3
"""
Test runner for Tetris RL project
Runs all unit tests and integration tests
"""
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
def run_all_tests():
    """Run all tests in the tests directory"""
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    return result.wasSuccessful()
def run_specific_test_module(module_name):
    """Run tests from a specific module"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(module_name)
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    return result.wasSuccessful()
def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) > 1:
        module_name = sys.argv[1]
        print(f"Running tests from module: {module_name}")
        success = run_specific_test_module(module_name)
    else:
        print("Running all tests...")
        success = run_all_tests()
    if success:
        print("\n All tests passed!")
        sys.exit(0)
    else:
        print("\n Some tests failed!")
        sys.exit(1)
if __name__ == '__main__':
    main() 