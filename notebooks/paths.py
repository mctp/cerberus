from pathlib import Path

def get_project_root() -> Path:
    """
    Return the project root directory by searching for pyproject.toml
    starting from this file's directory and moving up.
    """
    # Start from the directory where this utility file is located (notebooks/)
    current_path = Path(__file__).resolve().parent
    
    # Traverse up to find pyproject.toml
    while current_path != current_path.parent:
        if (current_path / "pyproject.toml").exists():
            return current_path
        current_path = current_path.parent
        
    # If not found (unlikely in this structure), return the location of this file's parent
    # which is likely the notebooks directory, or just fallback to cwd
    return Path.cwd()
