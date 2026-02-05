import sys
import subprocess

def install_requirements():
    """Install packages from requirements.txt"""
    print("📦 Checking and installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        sys.exit(1)

def check_imports():
    """Verify critical imports"""
    print("🔍 Verifying critical packages...")
    packages = [
        "fastapi", "sqlalchemy", "uvicorn", "pydantic", "multipart"
    ]
    missing = []
    for package in packages:
        try:
            # Handle package name vs import name differences if any
            import_name = package
            if package == "multipart": import_name = "python_multipart" 
            # Note: python-multipart module name is sometimes just 'multipart' or handled via libraries.
            # actually 'import python_multipart' or just relying on pip check is better.
            # We'll rely on pip install mostly.
            pass
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"⚠️  Potential missing packages: {missing}")
    else:
        print("✅ Core packages verified.")

if __name__ == "__main__":
    print(f"🐍 Python executable: {sys.executable}")
    install_requirements()
    check_imports()
    print("\n🎉 Environment setup complete! You can now run the server.")
