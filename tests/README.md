# Tetris RL Tests

This directory contains comprehensive unit tests and integration tests for the Tetris RL project with **117 total tests** covering all major components.

## Test Structure

### Unit Tests (105 tests)
- `test_game_data.py` - Tests for the core game logic (Shape and BoardData classes) - **17 tests**
- `test_tetris_env.py` - Tests for the Tetris environment wrapper - **19 tests**
- `test_dqn_agent.py` - Tests for DQN agent and other agent implementations - **17 tests**
- `test_model.py` - Tests for the neural network architecture - **17 tests**
- `test_preprocessing.py` - Tests for feature preprocessing utilities - **19 tests**
- `test_config.py` - Tests for configuration parameters - **16 tests**

### Integration Tests (12 tests)
- `test_integration.py` - Tests for component interactions and full system integration - **12 tests**

## Running Tests

### Quick Start

Run all 117 tests:
```bash
cd AppliedML-Tetris
python -m unittest discover tests -v
```

Expected output:
```
Ran 117 tests in ~5.0s
OK
```

### Using unittest (built-in)

Run all tests with verbose output:
```bash
cd AppliedML-Tetris
python -m unittest discover tests -v
```

Run specific test file:
```bash
python -m unittest tests.test_game_data -v
```

Run specific test class:
```bash
python -m unittest tests.test_game_data.TestShape -v
```

Run specific test method:
```bash
python -m unittest tests.test_game_data.TestShape.test_shape_initialization -v
```

### Using the custom test runner

Run all tests:
```bash
cd AppliedML-Tetris/tests
python run_tests.py
```

Run specific module:
```bash
cd AppliedML-Tetris/tests
python run_tests.py test_config
```

Expected output for successful run:
```
Running tests from module: test_config
...
----------------------------------------------------------------------
Ran 16 tests in 0.012s

All tests passed!
```

### Using pytest (if installed)

Install pytest:
```bash
pip install pytest
```

Run all tests:
```bash
cd AppliedML-Tetris
pytest
```

Run specific test file:
```bash
pytest tests/test_game_data.py -v
```

Run tests with coverage (if pytest-cov is installed):
```bash
pip install pytest-cov
pytest --cov=. --cov-report=html
```

Run only fast tests (excluding slow integration tests):
```bash
pytest -m "not slow"
```

## Test Categories

### Unit Tests (105 tests)
Test individual components in isolation:

**Game Logic (17 tests)**
- Shape rotation and coordinate calculation
- Board operations and piece placement
- Line clearing mechanics
- Game over detection

**Environment (19 tests)**
- Action handling and validation
- Reward calculation system
- Observation generation
- Episode management

**Agents (17 tests)**
- DQN agent behavior and learning
- Random and heuristic agents
- Epsilon decay and exploration
- Model saving/loading

**Neural Network (17 tests)**
- Architecture validation
- Forward pass functionality
- Gradient flow
- Dropout and training modes

**Preprocessing (19 tests)**
- Feature extraction from board states
- PCA dimensionality reduction
- Normalization and scaling
- Board analysis (holes, heights, wells)

**Configuration (16 tests)**
- Parameter validation
- Type checking
- Path construction
- Reward system configuration

### Integration Tests (12 tests)
Test component interactions:

**Environment-Agent Integration (4 tests)**
- Full game loops with different agents
- Multi-episode training simulation
- Epsilon decay during gameplay

**Preprocessing Integration (2 tests)**
- Feature extraction pipeline
- PCA integration with agents

**Full System Integration (4 tests)**
- Complete training simulation
- Save/load functionality
- Evaluation mode testing
- Various board state handling

**Error Handling (2 tests)**
- Invalid action handling
- Malformed state recovery

## Test Performance

- **Total Tests**: 117
- **Execution Time**: ~5 seconds
- **Success Rate**: 100% (all tests passing)
- **Coverage**: All major components and interactions

### Test Timing Breakdown:
- Unit tests: ~3 seconds
- Integration tests: ~2 seconds
- Fastest test: ~0.001 seconds
- Slowest test: ~0.5 seconds (neural network tests)

## Test Coverage Details

The tests comprehensively cover:

✅ **Core Game Mechanics**
- Piece movement, rotation, and placement
- Line clearing and scoring
- Game over conditions
- Board state management

✅ **AI Components**
- DQN agent training and inference
- Experience replay and target networks
- Multiple agent types (Random, Heuristic, DQN)
- Model persistence

✅ **Environment Interface**
- Gym-like API compliance
- Observation and action spaces
- Reward engineering
- Episode lifecycle

✅ **Feature Engineering**
- Board state analysis
- Height and hole calculations
- PCA dimensionality reduction
- Feature normalization

✅ **System Integration**
- End-to-end training pipelines
- Component interaction validation
- Error handling and recovery
- Configuration management

## Dependencies

The tests require:
- Python 3.7+
- PyTorch (for neural network tests)
- NumPy (for numerical operations)
- scikit-learn (for preprocessing tests)
- PyQt5 (for UI component tests)

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Make sure you're in the project root
cd AppliedML-Tetris
python -m unittest discover tests -v
```

**PyTorch Not Found**
```bash
pip install torch torchvision
```

**Slow Test Execution**
- Integration tests may take longer due to neural network operations
- Use `-m "not slow"` with pytest to skip slow tests during development

**Memory Issues**
- Some tests create temporary neural networks
- Tests automatically clean up temporary files
- Restart Python if memory usage becomes excessive

### Test Isolation

- Each test is designed to be independent
- Temporary files are automatically cleaned up
- Random seeds are controlled for reproducibility
- No persistent state between tests

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: Tetris RL Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests with unittest
      run: |
        python -m unittest discover tests -v
    
    - name: Run tests with pytest and coverage
      run: |
        pytest --cov=. --cov-report=xml --cov-report=html
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Development Workflow

### Adding New Tests

1. Create test file following naming convention: `test_<module>.py`
2. Inherit from `unittest.TestCase`
3. Use descriptive test method names: `test_<functionality>`
4. Include docstrings explaining what is being tested
5. Add both positive and negative test cases

### Test Guidelines

- Keep tests fast (< 1 second each)
- Use mock data when possible
- Test edge cases and error conditions
- Ensure tests are deterministic
- Clean up resources in tearDown methods

### Running Specific Test Suites

```bash
# Run only unit tests
python -m unittest tests.test_game_data tests.test_tetris_env tests.test_dqn_agent tests.test_model tests.test_preprocessing tests.test_config -v

# Run only integration tests
python -m unittest tests.test_integration -v

# Run tests for specific component
python -m unittest tests.test_dqn_agent -v
``` 