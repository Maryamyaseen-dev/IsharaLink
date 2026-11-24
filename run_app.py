import os
import sys

# Change directory to web_app
os.chdir(os.path.join(os.path.dirname(__file__), "web_app"))
sys.path.insert(0, os.getcwd())

# Import and run app
from app import app

if __name__ == "__main__":
    print("\n" + "="*60)
    print("   ISHARALINK - COMPLETE BIDIRECTIONAL SYSTEM")
    print("   Open your browser at: http://localhost:5000")
    print("="*60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
