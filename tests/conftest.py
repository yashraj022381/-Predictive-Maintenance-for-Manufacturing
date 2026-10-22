"""Runs before the tests: point them at a throwaway database."""
import os
import tempfile

os.environ["PM_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test_history.db")
