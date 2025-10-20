"""
Debug script for MeshCat animation issues with Python 3.11+
"""

import sys
print(f"Python version: {sys.version}")

# Try to patch the typing issue before importing meshcat
import typing
if not hasattr(typing, '_ClassVar'):
    print("Patching typing._ClassVar...")
    typing._ClassVar = typing.ClassVar

try:
    import meshcat
    print(f"MeshCat imported successfully! Version: {meshcat.__version__ if hasattr(meshcat, '__version__') else 'unknown'}")
    import meshcat.geometry as g
    import meshcat.transformations as tf
    print("MeshCat modules imported successfully!")
except ImportError as e:
    print(f"ImportError: {e}")
    print("MeshCat not installed. Install with: pip install meshcat")
    sys.exit(1)
except AttributeError as e:
    print(f"AttributeError during import: {e}")
    print("\nThis is likely a compatibility issue with Python 3.11+")
    print("Trying to diagnose the issue...")

    # Try to find which submodule is causing the issue
    try:
        import meshcat.geometry
        print("meshcat.geometry imported OK")
    except Exception as e2:
        print(f"meshcat.geometry failed: {e2}")

    try:
        import meshcat.transformations
        print("meshcat.transformations imported OK")
    except Exception as e2:
        print(f"meshcat.transformations failed: {e2}")

    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nAll imports successful! Testing visualizer creation...")

try:
    vis = meshcat.Visualizer()
    print(f"Visualizer created successfully!")
    print(f"URL: {vis.url()}")

    # Try basic operations
    vis.delete()
    print("delete() works")

    vis["test"].set_object(
        g.Sphere(0.1),
        g.MeshLambertMaterial(color=0xff0000)
    )
    print("set_object() works")

    vis["test"].set_transform(
        tf.translation_matrix([0, 0, 0])
    )
    print("set_transform() works")

    print("\nMeshCat is working correctly!")
    print(f"Visit {vis.url()} to see the test visualization")

except Exception as e:
    print(f"Error during visualizer test: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
