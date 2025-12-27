import sys
import requests
import threading
from PyQt6.QtWidgets import QApplication, QLineEdit, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette
import keyboard

# Create a worker class to handle signals from background threads
class SignalHandler(QObject):
    triggered = pyqtSignal()

class QuickBar(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.backend_url = "http://127.0.0.1:8000"
        
    def init_ui(self):
        # Window setup
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Geometry (Top Center)
        screen = QApplication.primaryScreen().geometry()
        width = 700
        height = 80
        x = (screen.width() - width) // 2
        y = screen.height() // 4
        self.setGeometry(x, y, width, height)
        
        # Layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Input Field
        self.input = QLineEdit(self)
        self.input.setPlaceholderText("Capture a thought... (Alt+Space to toggle)")
        self.input.returnPressed.connect(self.submit)
        self.input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(30, 30, 30, 240);
                color: #EEE;
                border: 1px solid #444;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 20px;
                font-family: 'Segoe UI', sans-serif;
            }
            QLineEdit:focus {
                border: 1px solid #777;
            }
        """)
        
        # Status Label (Hidden by default)
        self.status = QLabel("", self)
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet("color: #AAA; font-size: 10px;")
        
        layout.addWidget(self.input)
        layout.addWidget(self.status)
        
    def submit(self):
        text = self.input.text()
        if not text:
            self.hide()
            return
            
        # Optimistic UI: Hide immediately
        self.hide()
        self.input.clear()
        
        # Send to backend in background thread
        threading.Thread(target=self.send_to_backend, args=(text,), daemon=True).start()
        
    def send_to_backend(self, text):
        try:
            requests.post(f"{self.backend_url}/capture", json={"text": text})
            # Success (Sound or toast could go here)
        except Exception as e:
            # If backend is down, we might want to log it or save locally
            print(f"Failed to send note: {e}")
            with open("quickcapture_offline_log.txt", "a") as f:
                f.write(f"{text}\n")

    def show_bar(self):
        self.show()
        self.activateWindow()
        self.input.setFocus()
        self.input.selectAll()

def run():
    app = QApplication(sys.argv)
    bar = QuickBar()
    
    # Signal handler for thread-safe GUI updates
    handler = SignalHandler()
    
    def toggle():
        if bar.isVisible():
            bar.hide()
        else:
            bar.show_bar()
            
    handler.triggered.connect(toggle)

    # Register hotkey
    try:
        # Emit signal instead of calling toggle directly
        keyboard.add_hotkey('alt+space', handler.triggered.emit)
    except ImportError:
        print("Keyboard library not found or requires admin rights.")
    
    print("QuickCapture Launcher Running... Press Alt+Space")
    sys.exit(app.exec())

if __name__ == "__main__":
    run()
