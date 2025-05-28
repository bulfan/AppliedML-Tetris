import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from env.game_data import Shape, BoardData, BOARD_DATA
class TestShape(unittest.TestCase):
    """Test cases for the Shape class"""
    def setUp(self):
        """Set up test fixtures"""
        self.shape_i = Shape(Shape.shapeI)
        self.shape_o = Shape(Shape.shapeO)
        self.shape_t = Shape(Shape.shapeT)
        self.shape_none = Shape(Shape.shapeNone)
    def test_shape_initialization(self):
        """Test shape initialization"""
        self.assertEqual(self.shape_i.shape, Shape.shapeI)
        self.assertEqual(self.shape_o.shape, Shape.shapeO)
        self.assertEqual(self.shape_t.shape, Shape.shapeT)
        self.assertEqual(self.shape_none.shape, Shape.shapeNone)
    def test_shape_constants(self):
        """Test shape constants are correctly defined"""
        self.assertEqual(Shape.shapeNone, 0)
        self.assertEqual(Shape.shapeI, 1)
        self.assertEqual(Shape.shapeL, 2)
        self.assertEqual(Shape.shapeJ, 3)
        self.assertEqual(Shape.shapeT, 4)
        self.assertEqual(Shape.shapeO, 5)
        self.assertEqual(Shape.shapeS, 6)
        self.assertEqual(Shape.shapeZ, 7)
    def test_get_rotated_offsets(self):
        """Test rotation offset calculation"""
        offsets_0 = list(self.shape_i.getRotatedOffsets(0))
        offsets_1 = list(self.shape_i.getRotatedOffsets(1))
        self.assertEqual(len(offsets_0), 4)
        self.assertEqual(len(offsets_1), 4)
        offsets_o_0 = list(self.shape_o.getRotatedOffsets(0))
        offsets_o_1 = list(self.shape_o.getRotatedOffsets(1))
        self.assertEqual(offsets_o_0, offsets_o_1)
    def test_get_coords(self):
        """Test coordinate calculation"""
        coords = list(self.shape_i.getCoords(0, 5, 10))
        self.assertEqual(len(coords), 4)
        for coord in coords:
            self.assertIsInstance(coord, tuple)
            self.assertEqual(len(coord), 2)
    def test_get_bounding_offsets(self):
        """Test bounding box calculation"""
        minX, maxX, minY, maxY = self.shape_i.getBoundingOffsets(0)
        self.assertIsInstance(minX, int)
        self.assertIsInstance(maxX, int)
        self.assertIsInstance(minY, int)
        self.assertIsInstance(maxY, int)
        self.assertLessEqual(minX, maxX)
        self.assertLessEqual(minY, maxY)
class TestBoardData(unittest.TestCase):
    """Test cases for the BoardData class"""
    def setUp(self):
        """Set up test fixtures"""
        self.board = BoardData()
        self.board.clear()
    def test_board_initialization(self):
        """Test board initialization"""
        self.assertEqual(self.board.width, 10)
        self.assertEqual(self.board.height, 22)
        self.assertEqual(len(self.board.backBoard), 220)        self.assertEqual(self.board.currentX, -1)
        self.assertEqual(self.board.currentY, -1)
        self.assertEqual(self.board.currentDirection, 0)
        self.assertEqual(self.board.currentShape.shape, Shape.shapeNone)
    def test_get_data(self):
        """Test getting board data"""
        data = self.board.getData()
        self.assertEqual(len(data), 220)
        self.assertIsInstance(data, list)
        self.assertTrue(all(cell == 0 for cell in data))
    def test_get_value(self):
        """Test getting value at specific position"""
        value = self.board.getValue(0, 0)
        self.assertEqual(value, 0)
        value = self.board.getValue(9, 21)
        self.assertEqual(value, 0)
    def test_clear(self):
        """Test board clearing"""
        self.board.backBoard[0] = 1
        self.board.currentX = 5
        self.board.currentY = 10
        self.board.clear()
        self.assertEqual(self.board.currentX, -1)
        self.assertEqual(self.board.currentY, -1)
        self.assertEqual(self.board.currentDirection, 0)
        self.assertTrue(all(cell == 0 for cell in self.board.backBoard))
    def test_try_move(self):
        """Test move validation"""
        shape = Shape(Shape.shapeI)
        self.assertTrue(self.board.tryMove(shape, 0, 5, 10))
        self.assertFalse(self.board.tryMove(shape, 0, -1, 10))
        self.assertFalse(self.board.tryMove(shape, 0, 15, 10))
    def test_create_new_piece(self):
        """Test piece creation"""
        result = self.board.createNewPiece()
        if result:            self.assertNotEqual(self.board.currentX, -1)
            self.assertNotEqual(self.board.currentY, -1)
            self.assertNotEqual(self.board.currentShape.shape, Shape.shapeNone)
            self.assertIn(self.board.nextShape.shape, range(1, 8))
    def test_move_operations(self):
        """Test basic move operations"""
        if self.board.createNewPiece():
            initial_x = self.board.currentX
            initial_y = self.board.currentY
            self.board.moveLeft()
            self.assertLessEqual(self.board.currentX, initial_x)
            self.board.moveRight()
            self.assertGreaterEqual(self.board.currentX, initial_x - 1)
    def test_rotation(self):
        """Test piece rotation"""
        if self.board.createNewPiece():
            initial_direction = self.board.currentDirection
            self.board.rotateRight()
            expected_direction = (initial_direction + 1) % 4
            self.assertIn(self.board.currentDirection, [initial_direction, expected_direction])
    def test_game_over_detection(self):
        """Test game over detection"""
        self.assertFalse(self.board.gameOver())
        for x in range(self.board.width):
            self.board.backBoard[x] = 1
        self.assertTrue(self.board.gameOver())
    def test_remove_full_lines(self):
        """Test line removal"""
        for x in range(self.board.width):
            self.board.backBoard[x + (self.board.height - 1) * self.board.width] = 1
        lines_removed = self.board.removeFullLines()
        self.assertEqual(lines_removed, 1)
        for x in range(self.board.width):
            self.assertEqual(self.board.getValue(x, self.board.height - 1), 0)
    def test_merge_piece(self):
        """Test piece merging"""
        if self.board.createNewPiece():
            shape = self.board.currentShape.shape
            coords = list(self.board.getCurrentShapeCoord())
            self.board.mergePiece()
            self.assertEqual(self.board.currentX, -1)
            self.assertEqual(self.board.currentY, -1)
            self.assertEqual(self.board.currentShape.shape, Shape.shapeNone)
            for x, y in coords:
                if 0 <= x < self.board.width and 0 <= y < self.board.height:
                    self.assertEqual(self.board.getValue(x, y), shape)
class TestBoardDataGlobal(unittest.TestCase):
    """Test cases for the global BOARD_DATA instance"""
    def test_global_board_data_exists(self):
        """Test that global BOARD_DATA instance exists"""
        self.assertIsInstance(BOARD_DATA, BoardData)
    def test_global_board_data_properties(self):
        """Test global BOARD_DATA properties"""
        self.assertEqual(BOARD_DATA.width, 10)
        self.assertEqual(BOARD_DATA.height, 22)
if __name__ == '__main__':
    unittest.main() 