def test_example():
    """Basic test to verify pytest is working."""
    assert 1 + 1 == 2

class TestExampleClass:
    """Example test class."""
    
    def test_inside_class(self):
        """Test inside a class."""
        assert "hello".upper() == "HELLO"
