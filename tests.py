import numpy as np
from main import (
    initialize_parameters,
    activation_function,
    reward_function,
    placecells,
    update_position
)

def test_activation_function():
    """Test the activation function."""
    params = initialize_parameters()
    
    # Test 1: Check if activation is bounded by rho
    ua = np.random.randn(10)
    act = activation_function(ua, params)
    assert np.all(act >= 0), "Activation should be non-negative"
    assert np.all(act <= params['actor']['rho']), f"Activation should not exceed rho ({params['actor']['rho']})"
    
    # ... rest of test implementation ...

def test_reward_function():
    """Test the reward function."""
    params = initialize_parameters()
    
    # ... test implementation ...

def test_place_cells():
    """Test place cell activity."""
    params = initialize_parameters()
    
    # ... test implementation ...

def test_position_update():
    """Test position updates and wall collisions."""
    params = initialize_parameters()
    
    # ... test implementation ...

if __name__ == '__main__':
    # Run all tests
    test_activation_function()
    test_reward_function()
    test_place_cells()
    test_position_update()
    print("\nAll tests passed!") 