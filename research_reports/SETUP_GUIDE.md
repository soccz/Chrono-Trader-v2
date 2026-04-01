# Environment Setup Guide (macOS)

This guide explains how to set up the project environment on a new macOS machine.

### Prerequisites

- **Homebrew:** The macOS package manager. If not installed, open Terminal and run:
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```

### Step-by-Step Setup

1.  **Create Virtual Environment:**
    Open a terminal in the project root folder (`gan_t`) and run:
    ```bash
    python3 -m venv venv
    ```

2.  **Activate Virtual Environment:**
    ```bash
    source venv/bin/activate
    ```
    *(Your terminal prompt should now start with `(venv)`)*

3.  **Install TA-Lib System Dependency:**
    ```bash
    brew install ta-lib
    ```

4.  **Install Required Python Packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Setup Complete

Once these steps are complete, the environment is ready. You can now run the main application, for example:

-   **To train the models:** `python main.py --mode train`
-   **To get quick recommendations:** `python main.py --mode quick-recommend`
