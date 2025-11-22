
import subprocess
import sys
import time
import os
import webbrowser
from threading import Thread

def print_banner():
    print("\n" + "="*70)
    print("║" + " "*20 + "PORTFOLIO OPTIMIZER LAUNCHER" + " "*21 + "║")
    print("="*70 + "\n")

def check_dependencies():
    """Check if required packages are installed"""
    print("📦 Checking dependencies...")
    
    required = [
        'fastapi', 'uvicorn', 'streamlit', 'pandas', 
        'numpy', 'scipy', 'yfinance', 'plotly', 'requests'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed\n")
    return True

def check_files():
    print("📁 Checking files...")
    
    required_files = ['optimizer.py', 'main.py', 'frontend.py']
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print(f"❌ Missing files: {', '.join(missing)}")
        return False
    
    print("✅ All files present\n")
    return True

def start_backend():
    print("🚀 Starting Backend API on port 8000...")
    print("-" * 70)
    
    try:
        subprocess.run(
            [sys.executable, "main.py"],
            check=True
        )
    except KeyboardInterrupt:
        print("\n⏹️  Backend stopped")
    except Exception as e:
        print(f"\n❌ Backend error: {e}")

def start_frontend():
    """Start Streamlit frontend"""
    time.sleep(3)  # Wait for backend to start
    
    print("\n" + "="*70)
    print("🎨 Starting Frontend UI on port 8501...")
    print("-" * 70)
    
    # Open browser after a delay
    def open_browser():
        time.sleep(5)
        print("\n🌐 Opening browser...")
        webbrowser.open("http://localhost:8501")
    
    Thread(target=open_browser, daemon=True).start()
    
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "frontend.py"],
            check=True
        )
    except KeyboardInterrupt:
        print("\n⏹️  Frontend stopped")
    except Exception as e:
        print(f"\n❌ Frontend error: {e}")

def main():
    """Main launcher"""
    print_banner()
    
    # Pre-flight checks
    if not check_files():
        print("\n❌ Setup incomplete. Please ensure all files are present.")
        sys.exit(1)
    
    if not check_dependencies():
        print("\n❌ Dependencies missing. Run: pip install -r requirements.txt")
        sys.exit(1)
    
    print("="*70)
    print("🎯 Starting Portfolio Optimizer...")
    print("="*70)
    print("\nℹ️  Tips:")
    print("   • Backend API will start on http://127.0.0.1:8000")
    print("   • Frontend UI will open at http://localhost:8501")
    print("   • Press Ctrl+C to stop both services")
    print("   • API docs available at http://127.0.0.1:8000/docs")
    print("\n" + "="*70 + "\n")
    
    backend_thread = Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Start frontend in main thread
    try:
        start_frontend()
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("⏹️  Shutting down Portfolio Optimizer...")
        print("="*70)
        print("\n✨ Thanks for using Portfolio Optimizer!")
        print("💡 Run again with: python run_app.py\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("\nFor help, see README.md or run: python test_api.py")
        sys.exit(1)
