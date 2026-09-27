"""
app.py / main.py
===============================================================================
Universal Application Entrypoint for Streamlit & Cloud Hosting Platforms
Delegates execution directly to ui_app.main()
===============================================================================
"""

import sys
from ui_app import main

if __name__ == "__main__":
    main()
