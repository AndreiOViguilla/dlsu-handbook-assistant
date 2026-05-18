"""
Pytest configuration and shared fixtures
"""
import pytest


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear answer cache between tests to avoid cache hits affecting results"""
    try:
        from api import answer_cache, stats
        answer_cache.clear()
        stats["total_visits"] = 0
        stats["total_questions"] = 0
    except ImportError:
        pass
    yield
    try:
        from api import answer_cache
        answer_cache.clear()
    except ImportError:
        pass
