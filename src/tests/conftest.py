"""
Test fixtures and sample data.
"""

import pytest
from pathlib import Path
import tempfile


@pytest.fixture
def sample_txt_file():
    """Create sample .txt file."""
    content = "This is a test validation log.\nLine 2: Test passed.\nLine 3: All checks complete."
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        f.flush()
        path = Path(f.name)
    
    yield path
    
    if path.exists():
        path.unlink()


@pytest.fixture
def sample_log_file():
    """Create sample .log file."""
    content = """
    [INFO] 2024-12-01 10:00:00 - Test started
    [DEBUG] 2024-12-01 10:00:01 - Loading configuration
    [INFO] 2024-12-01 10:00:02 - Running validation tests
    [PASS] 2024-12-01 10:00:05 - CPU test passed
    [PASS] 2024-12-01 10:00:08 - Memory test passed
    [INFO] 2024-12-01 10:00:10 - All tests completed successfully
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
        f.write(content)
        f.flush()
        path = Path(f.name)
    
    yield path
    
    if path.exists():
        path.unlink()


@pytest.fixture
def sample_validation_report():
    """Create sample validation report."""
    content = """
    POST-SILICON VALIDATION REPORT
    ==============================
    
    Product: Test Chip v2.0
    Date: December 1, 2024
    Engineer: Test Engineer
    
    SUMMARY
    -------
    Overall Status: PASS
    Total Tests: 150
    Passed: 148
    Failed: 2
    
    DETAILED RESULTS
    ----------------
    
    1. CPU Core Tests
       - Core 0: PASS
       - Core 1: PASS
       - Core 2: PASS
       - Core 3: PASS
    
    2. Memory Tests
       - DDR4 Interface: PASS
       - ECC Tests: PASS
       - Bandwidth: 25 GB/s (Expected: 20 GB/s)
    
    3. Power Management
       - Idle Power: 5W (Expected: <10W)
       - Active Power: 95W (Expected: <100W)
       - Power Gating: PASS
    
    4. Known Issues
       - Issue #1: Clock skew on GPIO pins
       - Issue #2: USB 3.0 compliance marginal
    
    RECOMMENDATIONS
    ---------------
    - Investigate clock distribution for GPIO
    - Retune USB PHY parameters
    - All other tests meet specifications
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.report', delete=False) as f:
        f.write(content)
        f.flush()
        path = Path(f.name)
    
    yield path
    
    if path.exists():
        path.unlink()
